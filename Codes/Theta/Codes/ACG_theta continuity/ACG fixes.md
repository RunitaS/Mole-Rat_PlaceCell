# ACG fixes

- [ ] **Integration of FOOOF into ACG pipeline is pending. Problems with increasing the fit range from 1-40 Hz to 1-100 Hz.**

- [x] ** Add Bycycle theta +ve epochs to ACG pipeline** *Not required, instead use ED_min value threshold.

- [x] ** Is the signal cutoff at 100 Hz?** *no noticeable change*

- [x] ** run the ACG analysis on aperiodic denoised signal.** *currently 6-7 Hz has very few epochs.*

- [x] ** Final correlation values are pretty low (avg 0.2).**
*cause could be theta assymetry? check bycycle pipeline to see if it's giving wrong values of theta symmetry.*
*general values are low so adding a threshold will lead to rejection of too many epochs. Don't add peakrangenorm criteria. You'll encounter same problem as fBOSC and Bycycle.*

- [x] ** Get criteria for acceptable values of peakrangenorm. Also and indiciator or fit quality.** *0 to 1 range with 0 indicating bad fit and 1 indicating perfect fit.* 
*check if low ED_min value correlates with/causes low peakrangenorm value.*
*<FIX: Doesnt matter. They are linearly correlated so having a threhold on ED_min is enough. It is easier to determine the threshold value for ED_min anyway. Correlation value threshold will be arbitrary. ED_min value of 1 and above is known to have sinusoid phase offset.>*
<Compare ED_min vs peakrangenorm as a quality metric to quantify theta continuity, club both or use only 1? 'peakrangenorm' preffered over ED_min>


- [x] ** Get epochs with no sin fit. Probably flagged as NAN.** *same as ED_min >= 1*

- [x] ** Quantify ACG matched ref. in different frequency bins.**

- [x] ** For rejeted values of ED_min (which are above 1) get the freq_est to check if certain freqs are rejected more than others.**

- [x] ** Add ED_min threshold. Make it data driven instead of a fixed hard coded value.**
        *Check for ways to calculate the ED_min value when the Euclidean distance obtained from the mathcing ref. sinusoid is completely out of phase.*
        <FIX: value of ED_min = 1 usually means sin is out of phase. Verify this though. There might be a better way to cacluate the ED_min threshold>

- [x] ** Check if mech noise is causing the ed_min filter to reject the epoch. Use 8477's data for sanity check.**

- [ ] **Check what is causing the thick lines at the boundaries of centroid plots. Present in both immobility and mobility data. More prominent in 1059's data. Probably noise epochs.**