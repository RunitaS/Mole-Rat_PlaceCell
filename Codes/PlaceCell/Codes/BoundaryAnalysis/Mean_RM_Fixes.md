# Mean rate map and population analysis Fixes

- [x] ** Change smoothing from triangular kernel to Gaussian kernel.**

- [ ] **Use only place fields to consturct the mean RM. Discard all other bins.**

- [x] ** Checkcode for minimum session duration threshold. Some 40sec threshold added somewhere in the code.** *No such condition found anywhere in the codes.*

- [ ] **Add occupancy criteria to open arena. If center is undersampled, don't use the recording session for mean RM.**

- [ ] **Decide if you want to use Raw or SMoothed RM for mean RMs.**

- [x] <Major fix:> ** Use pass index method to normalize rate maps of individual sessions.** *Alt, check Muessig's method for data nomralization.*
*Figure out a method to handle NAN bins. Calculate weighted means.*

- [x] ** 1D Circ data was linearized in the code. Bin size is 4cm. Change to 2D.**

- [x] ** Check the alignments of each quadrant. Plot the method of overlaps.** *Something weird is happening with open field overlapping.*

- [ ] **Do Population vector analysis to see if bins are more concentrated in a specific direction. Also compare between magnetic and topographic orientations.**

- [x] ** Add remaining parameters to boundary rep. characterization.** *Mean firing rate and stability used in Muessig et.al., might as well add all other comparision parameters.*>