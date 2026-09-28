 All percentage metrics still use the chi-square test. I checked the new code on made-up data: normal data went to ANOVA and t-tests, skewed data went to Kruskal-Wallis and Mann-Whitney U, and a two-group case used Mann-Whitney U directly. I haven't run it on your real workbook.

What the code used before
All continuous metrics (peak firing rate, spatial information, sparsity, coherence, stability, both speed scores, mean firing rate, number of fields, field area and % area) got the same treatment:

Normality (Shapiro-Wilk) and equal variance (Levene) were run but only printed. They didn't affect which test was used.
Kruskal-Wallis across the three arenas, then pairwise Wilcoxon rank-sum tests with Holm-Bonferroni correction.
The reason was presumably caution: these metrics are usually skewed or bounded, so rank-based tests are always safe. Wilcoxon rank-sum gives essentially the same p-values as Mann-Whitney U.

What the code does now
In compare_groups, each arena group gets a Shapiro-Wilk test (a group with fewer than 3 cells counts as non-normal):

Outcome	Test across 3 groups	Pairwise post-hoc
All groups normal	One-way ANOVA (F)	t-test: Student's if Levene p ≥ 0.05 (checks for similarity of variance), otherwise Welch's
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


## Stats v2: LMM

Your arena data has more repetition than you thought
Each session is a separate row, so a cell recorded in 3 sessions of a day is counted 3 times within its arena:

Place-cell rows	Unique cells	Cells in more than one session
Linear	332	195	95
Open	309	235	62
Circle	75	75	0 (see caveats)
About 10 cells also match across arenas on the same day, but that small overlap isn't the main issue. So both scripts have the same problem: partly paired data. Friedman can't handle cells that are missing from some sessions, and Kruskal–Wallis wrongly assumes every row is an independent cell.

What I recommend, and what the new code does
A cell is identified as animal + arena + day + unit label, since each day's sessions were spike-sorted together.

Outcome	Omnibus test	Post hoc
Continuous metrics	Rank-based mixed model with a random intercept per cell (likelihood-ratio χ²)	Pairwise z-tests, Holm-corrected
Percentages	Logistic GEE with cells as clusters and cluster-robust standard errors (Wald χ²)	Pairwise odds ratios, Holm-corrected
Speed direction (p / n / non)	Multinomial GEE, clustered by cell	Pairwise 2-df Wald tests, Holm-corrected
Rank-based mixed model: it compares groups on ranks like Kruskal–Wallis, so skew doesn't matter. Each repeated cell is compared with itself, and cells recorded only once still count.
GEE (generalized estimating equations): a logistic regression whose standard errors allow for one cell contributing several yes/no rows.
Fallback: if no cell is repeated, the rows really are independent, so the code uses the classic tests (Kruskal–Wallis with Mann–Whitney, or χ² with Fisher's exact). It does the same if a group sits at 0% or 100%, because the GEE can't be estimated then. A new model_note column and the plot captions say which test was used and why.
The pairing matters: Open sparsity, Cntrl vs Rotate, went from p(Holm) = 0.28 under Kruskal–Wallis to 0.036. A paired Wilcoxon signed-rank test on the 12 cells recorded in both sessions agrees (p = 0.002).
Post-hoc tests for your percentage (chi-square) comparisons
Your current code does this:

Omnibus: a χ² test of independence on the groups × (yes/no) table. It compares observed counts with what you'd expect if the percentage were the same in every group. A significant result only tells you that at least one group differs.
Post hoc, to find which groups differ:
For yes/no outcomes, it runs Fisher's exact test on each 2×2 sub-table (group A vs group B × yes/no). Fisher's test gives an exact p-value, so it stays valid when some counts are small (under 5), where the χ² approximation breaks down. It also reports an odds ratio as the effect size.
For speed direction, it runs a 2×3 χ² test for each pair of groups.
Holm–Bonferroni correction across the pairwise tests. With 3 groups there are 3 comparisons, so without correction the chance of at least one false positive is about 14% instead of 5%. Holm sorts the p-values and multiplies the smallest by 3, the next by 2 and the last by 1, keeping the adjusted values in order. It controls the error rate like Bonferroni but has more power.
Every one of these tests assumes each row is an independent cell, which is why I swapped them for the GEE versions.

Files
CellClusteredStats.py: new shared stats module
SessionType_StatsComparison_v5.py and ArenaType_StatsComparison_v6.py: new versions; the old ones are unchanged
Plots, sheets and workflow are the same as before. The descriptives now also have an n_cells column. Run them in the tetrode environment, which has statsmodels.

Caveats
The arena comparison is confounded with animal. Linear cells come only from Fa1059 and Fa23BD, and Circle cells only from Fa5834 and Fa8477. No statistical model can separate "arena effect" from "animal difference" here, so say so in the thesis, or restrict the comparison to Open vs the other arena within the same animals.
Circle (Fa8477) was sorted per session (unit names like TT1_0001_SS_01), so its cells can't be matched across sessions and are treated as independent. If they're really the same cells, Circle p-values are somewhat too optimistic. Sorting a day's Circle sessions together would fix this.
LINK_CELLS_ACROSS_ARENAS = True in the arena script treats the same unit label on the same day in two arenas as one cell. That's only correct if both arenas were sorted together; otherwise set it to False.