# Rayleigh and Observed Null Algos

## Algo for Rayleigh test:

Add a rayleigh test on the mean rate maps and the KDE postivie clusters only for open field data. 
So in total there will be 3 rate maps compared (Overall mean rate map, proportion of peaks rate map, 
field only mean rate map, singificant KDE clusters for overall mean rate maps, significant KDE clusters 
for proportion of peaks rate map, significant KDE clusters for field only mean rate maps).

The algorithm to carry out the test is as follows:

1. Normalize the distance from the center of the open field to the walls, with 0 at the center of the 
60cm diameter field and 1 at the periphery/walls. So the bins around the center will get a multiplier 
factor of 0 and the bins near the wall will get a multiplier factor of 1. 

2. For the type of rate map being analyzed, whichever bin is being considerd, use the multiplier obtained 
for that bin with respect to distance from the wall. 

3. Divide all six rate maps is angular bins, 12 degrees wide. Distribute all the x,y position bins into 
these angular bins, make sure that no bin is repeated in more than 1 angular bin. Calculate the average 
magnitude value obtained from each bin considered in that angular bin post modification of firing rate value 
that incorporates the distance to wall multiplier. I would like to control for higher number of samples 
as we move towards the wall just because the area increases as we move from cener towards the wall, hence 
increasing the number of bins, how do I do this?

4. On the magnitude values for each angular bin, perform a Rayleigh vector test. Make sure you also test 
for bimodality. BEcause some of my plots have a bimodal distribution. Give stats results for a Rayleigh test 
for unimodal distributuion and bimodal distribution.


# CORRECT OBSERVED NULL ALGO:

I want the observed null to be is what a mean rate map would look like if the animals travelled through 
the trajectory that did (data for this is available in the tracking files.  If all animals had place fields 
that were uniformly covering the arena i.e. tiling the arena under consideration unfirmly and also not 
overlapping with each other at any bin, and the animals had travelled through this kind of spatial map, what 
would the mean overall rate map, mean field only rate map, and peak proportion rate map look like. 

Here's the algrithm for this process. 
Step 1:
Get the occupancy maps from all the tracking files, concatenate all the data as mentioned in the mean occupancy 
code attached.

Step 2: 
For each arena (Open, Linear, Circle), Get simulated sets of perfect place fields (firing rate should use the 
pass index method: peak should have rate 1, with neightbouring bins gradually decreasing their firing rate upto 0). 
Tile the entire arena uniformly with these place fields. They should not over lap with each other at any bin. 
For now set the field size as 3 by 3 bins so it will have 9 contiguous bins in total. Increase the number of bins 
by 1 to 3 bins if required so that it fits the geometry to make it uniform and non overlapping. 

Step 3: 
Consider an animal travelled this cognitive map and fired at these locations at the rate predicted by the fields 
tiling the arena unifmormly with the trajectory extracted from the concatenated trajectory path. what would be 
the resultant mean overall rate map? resulatant mean field only rate map (which includes only bins with firing 
rate upto 20% of the peak bin firing rate hence some bins will be excluded from this)? resultant peak proportion map?
Get these 3 types of final rate maps which will be the observed nulls for their corresponding mean rate map type.

Step 4: 
Apply KDE analysis to the observed null maps (same as the one used on the real data).

Step 5:
Compare the Kernels of real mean maps with their corresponding observed null kernels using Duong's test.