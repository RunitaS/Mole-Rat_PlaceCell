# Occ Null: Method 2

take each of the 3 mean rate map, and shuffle the bins randomly. 
perform KDE analysis on the shuffled mean rate maps.
Compare the Kernels of real vs shuffled mean rate maps for all three types.
Repeat this procedure 1000 times.
Check for regions of the arena where there was significant difference in the Duong's test in either directions more than 95% of the time. SO the confidence interval will be from 2.5% to 97.5% to cotrol for both reduction and increase in place encoding in a particular region.

## Execution

Randomly shuffles the map's values across the bins the cells sampled.
Fits the same KDE as the real map, with the same domain and edge correction.
Runs the Duong test of real against shuffled.
Repeats 1000 times and records, per bin, the fraction of shuffles where real was significantly higher and the fraction where it was significantly lower.
Marks a bin green (real higher) or magenta (real lower) if it was significant in the same direction in at least 95% of shuffles.
A choice for you: "more than 95% of the time" and "2.5% to 97.5%" can mean two different thresholds. I used 95% per direction. If you meant that the whole 95% range must exclude "no difference", set SHUFFLE_CONSISTENCY = 0.975.

I also added the classic percentile version of the 2.5–97.5% idea as a separate panel. It marks bins where the real density is above the 97.5th or below the 2.5th percentile of the 1000 shuffled densities.

Results (open field, 55 cells)

Map	Consistently higher (green)	Consistently lower (magenta)	Percentile envelope (above / below)
Overall	none (0 significant bins in any shuffle)	none	170 / 242 bins
Field-only	none (at most 20% of shuffles)	1 cluster of 4 bins, 19 cm from the wall, 98% of shuffles	181 / 184 bins
Peak proportion	none	none	31 / 30 bins
How to read this

This is not an occupancy control. Shuffling removes the spatial structure but not the dwell time, so the shuffled KDE is flat over the sampled area. The test asks whether mass is clustered compared with a spatially random map, not whether there is more than the animal's sampling explains; Method 1 tests that.
The 1000 repeats change little. Every shuffled KDE comes out nearly flat, so each bin is significant in close to 0% or 100% of shuffles. The result is close to a single Duong test against a uniform density.
The two criteria disagree. The Duong test finds almost nothing because it accounts for the uncertainty in the real KDE. The percentile envelope flags most of the arena because it only reflects the tiny spread among shuffles, and it isn't corrected for testing hundreds of bins at once. I'd treat the envelope as descriptive, not as evidence.
Methods 1 and 2 agree. Neither finds real data significantly higher near the walls; the only significant region in both is a small "lower than null" patch in the field-only map, in the arena interior.
Outputs: ShuffleDuong_<kind>.png (one per map type), ShuffleDuong_Clusters.xlsx and ShuffleDuong_Results.npz, plus the usual real mean-map and KDE figures.