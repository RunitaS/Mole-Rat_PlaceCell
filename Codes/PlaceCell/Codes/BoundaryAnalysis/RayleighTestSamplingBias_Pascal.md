# Rayleigh Test Sampling Bias

We want to add a rayleigh test on the mean rate maps for open field data. We have 3 types of mean rate maps compared (Whole mean rate map, field only mean rate map, proportion of peaks rate map).

The algorithm to carry out the test is as follows:

1. Normalize the distance from the center of the open field to the walls, with 0 at the center of the 60cm diameter field and 1 at the periphery/walls. So the bins exactly at the center will get a multiplier factor of 0 and the bins near the wall will get a multiplier factor of 1. 

2. For the type of rate map being analyzed, for every bin in that mean rate map, get the multiplier obtained for each bin with respect to distance from the wall. Multiple the firing rate with this multiplier factor.

3. Divide all three mean rate maps is angular bins, 12 degrees wide (Not sure if we should switch to 36 degrees though?). Distribute all the x,y position bins into these angular bins, ensuring that no bin is repeated in more than 1 angular bin. 

4. Calculate the average magnitude value obtained from each bin considered in that angular bin post modification of firing rate value (incorporates the distance to wall multiplier). 

**Sampling bias: we need to control for higher number of samples as we move towards the wall just because the area increases as we move from center to wall, hence increasing the number of bins, how do we do this?**

5. On the magnitude values for each angular bin, perform a Rayleigh vector test. Make sure you also test for bimodality. Because some of the mean rate maps plots multiple peaks around the walls.