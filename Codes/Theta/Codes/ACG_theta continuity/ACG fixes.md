# ACG fixes

- [ ] **Add Bycycle theta +ve epochs to ACG pipeline**

- [x] ** Add ED_min threshold. Make it data driven instead of a fixed hard coded value.**
        *Check for ways to calculate the ED_min value when the Euclidean distance obtained from the mathcing ref. sinusoid is completely out of phase.*
        <FIX: value of ED_min = 1 usually means sin is out of phase. Verify this though. There might be a better way to cacluate the ED_min threshold>

- [x] ** Check if mech noise is causing the ed_min filter to reject the epoch. Use 8477's data for sanity check.**

- [ ] **Check what is causing the thick lines at the boundaries of centroid plots. Present in both immobility and mobility data. More prominent in 1059's data. Probably noise epochs.**