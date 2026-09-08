# Adaptive binning fixes

- [ ] **Peak Fr and SIS values are way off.**

- [ ] **All repo codes have pixel based implementation. Adapt it to cm. Smoothing will vary based on same alpha used for cm tracking values.**

- [ ] **Jim's code doesn't seem to be behavior controlled. It's both occupancy and spike controlled making it adaptive smoothing instead of adaptive binning.** *the criterion formula used is different from Skaggs formula. Jim: α²·occ²·r²·nspikes² > 1*

- [ ] **Check if SIS is in bits/spike or bits/sec.**

- [ ] **Adpt binning uses physical arena size as an input. Check if that is causing inflation of values.**