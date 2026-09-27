 All percentage metrics still use the chi-square test. I checked the new code on made-up data: normal data went to ANOVA and t-tests, skewed data went to Kruskal-Wallis and Mann-Whitney U, and a two-group case used Mann-Whitney U directly. I haven't run it on your real workbook.

What the code used before
All continuous metrics (peak firing rate, spatial information, sparsity, coherence, stability, both speed scores, mean firing rate, number of fields, field area and % area) got the same treatment:

Normality (Shapiro-Wilk) and equal variance (Levene) were run but only printed. They didn't affect which test was used.
Kruskal-Wallis across the three arenas, then pairwise Wilcoxon rank-sum tests with Holm-Bonferroni correction.
The reason was presumably caution: these metrics are usually skewed or bounded, so rank-based tests are always safe. Wilcoxon rank-sum gives essentially the same p-values as Mann-Whitney U.

What the code does now
In compare_groups, each arena group gets a Shapiro-Wilk test (a group with fewer than 3 cells counts as non-normal):

Outcome	Test across 3 groups	Pairwise post-hoc
All groups normal	One-way ANOVA (F)	t-test: Student's if Levene p ≥ 0.05, otherwise Welch's
Any group non-normal	Kruskal-Wallis (H)	Mann-Whitney U (two-sided)
Only 2 groups present	none; the t-test or Mann-Whitney U is used directly	—
All pairwise p-values are Holm-corrected, as before. The plot text and the Excel test column now show which test was used for each metric (F, H, t or U). The same logic also applies to the pooled speed-score histogram stats.

Caution: Shapiro-Wilk becomes very sensitive with large groups (roughly 50+ cells per arena). Small departures from normality will fail it, so expect most of your metrics to end up on the Kruskal-Wallis / Mann-Whitney U path.

The post-hoc test after the chi-square test
Every percentage comparison first runs a chi-square test of independence on the arena × outcome count table (compare_categorical and compare_speed_direction). The post-hoc test then depends on the outcome:

Yes/no outcomes (place-cell yield, theta modulation, coherence shuffle significance, stability significance, both speed-modulation tests): pairwise Fisher's exact tests on each 2×2 table (arena A vs B × yes/no), Holm-corrected across the 3 pairs. The plots report an odds ratio for each pair.
Speed direction (p-Speed / n-Speed / non-speed): pairwise chi-square tests on each 2×3 table (arena A vs B × 3 categories), Holm-corrected.
Two things to be aware of:

Post-hoc tests run and are plotted even when the overall test is not significant. Strictly, you should only interpret them when it is.
The code doesn't check the chi-square rule that expected counts should be at least 5. That rule matters for the speed-direction table, where n-Speed counts may be small.
Summary of all tests
Tests across the three arenas: one-way ANOVA or Kruskal-Wallis for continuous metrics; chi-square test of independence for percentages.
Post-hoc tests: Student's or Welch's t-test, Mann-Whitney U, Fisher's exact, and pairwise chi-square.
Multiple-comparison correction: Holm-Bonferroni, applied to the 3 pairwise tests within each metric, not across metrics.
Checks: Shapiro-Wilk decides parametric vs non-parametric; Levene decides Student's vs Welch's t-test.