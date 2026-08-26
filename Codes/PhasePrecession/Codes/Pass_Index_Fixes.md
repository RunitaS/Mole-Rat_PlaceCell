# Pass Index Fixes

## Fies to-do:

- [ ] **Fix the <ERROR (Filter not stable due to sum(a) == 0, i.e., having a pole at z = 1!)>**

- [ ] **Fix the Error <TT8_SS_15_SS_01: ERROR (Digital filter critical frequencies must be 0 < Wn < 1)>** 

- [ ] **Check values of hilbert transformed phases of pass index to verify skewed blobby/multip peak fields**

- [ ] **When using LFP data from a tetrode that's not same as spike data, for all place cells from this tetrode, calculate te common offet for peak firing from trough of theta.**

- [ ] **Value of Φ0 will give the y intercept which mentions the preffered phase of theta. If not centered around 180, this can be problematic?**

- [ ] **Fix the tracking and output directory location. Tracking same as .ntt and .ncs., add hte cm/pixel choice code. Output also same as input directory.**
*Merge with your rate map plotting if required. You need to work on that code anyway.*

- [ ] **Current code requires a reference channel for LFP data. My data is already referenced at rec level**

- [ ] **look into the grid mode in <field_index_map>. How does it determine different fields? Do we need to provide this info before hand? Is it determines multifields, what algo is used for this?**
NOTE: grid: fixed band tuned to typical grid spacing. *Make sure you change this to irregular spacing*

- [ ] **Add Max 50msec ts distance criteria to spike_ts and pos_ts matching** 

- [ ] **Make sure that spikes from cluster 0 are skipped.**