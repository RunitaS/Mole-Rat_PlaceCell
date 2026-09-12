# Mean rate map and population analysis Fixes

- [x] <Major fix:> ** Circular track quadrant folds are aligned wrong. Use mirroring method instead of angular rotations.**

- [ ] **Edge vs Center analysis is run on smoothed Rm. Use the same for Mean RM peak map. Don't run on raw RM.** *Check Gaussian smoothing factor. Keep it consistent across all analyses.*

- [x] ** 1D Circ is converted to angular bins for some god forsaken reason. Convert it back to linear x,y bins.** *Probably why you get single bin.*

- [x] ** Change smoothing from triangular kernel to Gaussian kernel.**

- [x] ** Use only place fields to consturct the mean RM. Discard all other bins.** *Doesn't highlight the main point. Fields are evenly dispersed. Normalizing it is undoing the boundary effect you're tryna study. Try without normalization and check what Muessig has done in their code.*

- [x] ** Add coverage criteria for use of RM in Mean RM. Muessig et.al. used 94% coverage crtieria.** *You won't have to discard open field if you use a decent criteria. Your sample size will decrease but the analysis will be statistically testable.*

- [x] ** Checkcode for minimum session duration threshold. Some 40sec threshold added somewhere in the code.** *No such condition found anywhere in the codes.*

- [ ] **Decide if you want to use Raw or SMoothed RM for mean RMs.**

- [x] <Major fix:> ** Use pass index method to normalize rate maps of individual sessions.** *Alt, check Muessig's method for data nomralizat/width.*ion.*
*Figure out a method to handle NAN bins. Calculate weighted means.*

- [x] ** 1D Circ data was linearized in the code. Bin size is 4cm. Change to 2D.**

- [x] ** Check the alignments of each quadrant. Plot the method of overlaps.** *Something weird is happening with open field overlapping.*

- [ ] **Do Population vector analysis to see if bins are more concentrated in a specific direction. Also compare between magnetic and topographic orientations.**

- [x] ** Add remaining parameters to boundary rep. characterization.** *Mean firing rate and stability used in Muessig et.al., might as well add all other comparision parameters.*>