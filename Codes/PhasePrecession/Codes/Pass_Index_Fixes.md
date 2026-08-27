# Pass Index Fixes

## Fies to-do:

- [x] ** Fix the <ERROR (Filter not stable due to sum(a) == 0, i.e., having a pole at z = 1!)>**
*Fixed: `bandpass_filter` now designs via SOS (`output='sos'` + `sosfiltfilt`) instead of b/a + `filtfilt`, which was numerically unstable for narrow/near-DC bands.*

- [x] ** Fix the Error <TT8_SS_15_SS_01: ERROR (Digital filter critical frequencies must be 0 < Wn < 1)>**
*Fixed: `bandpass_filter` now clamps Wn into (0,1) relative to Nyquist and raises a clear ValueError if the band is degenerate, instead of letting scipy throw. Also switched `METHOD` from 'grid' to 'place' since this data is place cells -- the 'grid' auto filter band is a fixed constant tuned for grid-cell spacing and was mismatched to these fields/position units, which was likely producing the out-of-range bands in the first place.*

- [ ] **Add condition for theta hase succession. Is the code controlled for entry positon?**

- [ ] **Merge with theta modulation test code.**

- [ ] **Check values of hilbert transformed phases of pass index to verify skewed blobby/multip peak fields**

- [ ] **When using LFP data from a tetrode that's not same as spike data, for all place cells from this tetrode, calculate te common offset for peak firing from trough of theta.**

- [ ] **Value of Φ0 will give the y intercept which mentions the preffered phase of theta. If not centered around 180, this can be problematic?**

- [x] ** Fix the tracking and output directory location. Tracking same as .ntt and .ncs., add hte cm/pixel choice code. Output also same as input directory.**
*Fixed: `ROOT_FOLDER` is now recursively searched for every subfolder containing both .ncs and .ntt files (`find_session_folders`), each is processed independently (`process_session`), tracking .csv is auto-detected from that same session folder, and output is written to `<session_folder>/PhasePrecession_PassIndex`. Tracking now reads the pre-computed cm columns (D/E) directly from the .csv instead of pixel columns (B/C), so no pixel/cm choice is needed anymore -- `PIXELS_PER_CM` was removed.*

- [ ] **Current code requires a reference channel for LFP data. My data is already referenced at rec level**

- [ ] **look into the grid mode in <field_index_map>. How does it determine different fields? Do we need to provide this info before hand? Is it determines multifields, what algo is used for this?**
NOTE: grid: fixed band tuned to typical grid spacing. *Make sure you change this to irregular spacing*

- [ ] **Add Max 50msec ts distance criteria to spike_ts and pos_ts matching** 

- [x] **Make sure that spikes from cluster 0 are skipped.**
*Fixed: `load_ntt_spike_times` now unconditionally drops cell_number 0 (unsorted/noise), instead of only dropping it when other clusters were also present.*


*If hippocampus has both predctive and retrospective coding, why doesn't it have both theta phase precession and phase succession?*