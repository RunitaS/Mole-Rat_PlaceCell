# Theta Mod Prec Fixes

- [ ] **<Major fix: Add only significant theta epochs for phase precession>**

- [x] < Major fix:> ** Stats are wrong. Steeper slopes show no signif phase precession. flat ones do. Mostly dpendent on spike numbers.**

Bug 1 (primary cause of your symptom): anglereg's slope search tried only two Nelder-Mead starting points on a highly multimodal objective (many aliased local maxima), unbounded. For any real precession slope beyond near-zero, it frequently converged to an essentially arbitrary slope — confirmed in simulation (true slope 1.0 → fitted −14.3, true slope 2.0 → fitted −10.2, etc.). Flat cells were the one regime it reliably got right, which is exactly why flat-looking cells passed and steep ones didn't.
→ Fixed: replaced with a bounded coarse-grid search (±4 cycles/unit over the observed pass-index range) followed by local refinement at the best grid point.

Bug 2 (compounding, on top of bug 1): kempter_lincirc computed the fitted phase as mod(s*x, 2π) instead of mod(2π*s*x, 2π) — missing the same 2π factor the optimizer itself uses. This decoupled the reported ρ/p from the actual fitted slope for anything but tiny slopes, so even a correctly-fit steep cell could come back non-significant.
→ Fixed: added the missing 2π.

Bug 3 (secondary, per your go-ahead): slope_deg_per_pass used rad2deg(2π·s), which is degrees per 1 unit of pass index, not per the full pass (pass index spans 2 units, −1 to +1) as the docstring/classification logic intended — under-reporting by 2×.
→ Fixed: now rad2deg(4π·s).

*Verified the fix against simulated spikes with known ground-truth slopes: slope recovery went from essentially random to accurate, and significance now tracks true signal strength/spike count instead of being inverted. This will change which cells your pipeline calls significantly precessing/recessing (previously-flagged "flat but significant" cells will likely lose significance, and genuinely steep cells should gain it) — you'll want to re-run the pipeline to regenerate theta_phase.xlsx before drawing conclusions from it. I left ThetaModPrecFixes.md's corresponding TODO item unchecked since I didn't touch that file — let me know if you'd like me to check it off.*

- [x] < Major fix:> ** Change field threshold from 10% to 20%.**

- [ ] **Cross check if pass index code is executed according to Climer's code. Description in MEC paper does not match python code. Could be because of method used for grid cells,**

- [ ] **Currently there are considerable number of phase processing cells. Check for forward vs backward phase precession to remove the possibility of these being entry through butt.** *Not doable until Nauman is done with his analysis.*

- [ ] **Neurons could be theta phase rolling instead of processing.** *Net phase relationship is different from instantaneous relationship.*
<Cross check with Aditi's phase precession code>

- [ ] **Cross check procesion code with Hemanya's code.**

- [ ] **Go through the condition for significant slope**



## Pass Index Fixes

- [x] ** Look into the pass index +1 to -1 transiton problem.**
*NOTE: explanation at the bottom*

- [x] ** Fix the <ERROR (Filter not stable due to sum(a) == 0, i.e., having a pole at z = 1!)>**
*Fixed: `bandpass_filter` now designs via SOS (`output='sos'` + `sosfiltfilt`) instead of b/a + `filtfilt`, which was numerically unstable for narrow/near-DC bands.*

- [x] ** Fix the Error <TT8_SS_15_SS_01: ERROR (Digital filter critical frequencies must be 0 < Wn < 1)>**
*Fixed: `bandpass_filter` now clamps Wn into (0,1) relative to Nyquist and raises a clear ValueError if the band is degenerate, instead of letting scipy throw. Also switched `METHOD` from 'grid' to 'place' since this data is place cells -- the 'grid' auto filter band is a fixed constant tuned for grid-cell spacing and was mismatched to these fields/position units, which was likely producing the out-of-range bands in the first place.*

- [x] ** Add condition for theta hase succession. Is the code controlled for entry positon?** *It's called procession not succession.*

- [x] ** Merge with theta modulation test code.**

- [x] ** Check values of hilbert transformed phases of pass index to verify skewed blobby/multip peak fields**

- [ ] **When using LFP data from a tetrode that's not same as spike data, for all place cells from this tetrode, calculate te common offset for peak firing from trough of theta.** 
*Precession should be from 250 to 420, procession should be from 80 to 230 degrees of theta phase. Look for max spiking and match that with peak and trough of theta. The average shifting window can be used to calculate offset. Calc avg across all phase precession/processing cells on that dat form that TT.*
<Preffered solution: Calculate the phase difference between noisy and nearest clean neighbour from uber clean LFP epochs, offset the clean LFP by that amount and use this new LFP signal for spike-LFP analysis. Pad the signal for the offset time window.>

- [x] < Add plot with spikes overlayed on theta signal to visually inspect the preffered phase of spikes.>

- [ ] **Value of Φ0 will give the y intercept which mentions the preffered phase of theta. If not centered around 180, this can be problematic?**

- [x] ** Fix the tracking and output directory location. Tracking same as .ntt and .ncs., add hte cm/pixel choice code. Output also same as input directory.**
*Fixed: `ROOT_FOLDER` is now recursively searched for every subfolder containing both .ncs and .ntt files (`find_session_folders`), each is processed independently (`process_session`), tracking .csv is auto-detected from that same session folder, and output is written to `<session_folder>/PhasePrecession_PassIndex`. Tracking now reads the pre-computed cm columns (D/E) directly from the .csv instead of pixel columns (B/C), so no pixel/cm choice is needed anymore -- `PIXELS_PER_CM` was removed.*

- [x] ** Current code requires a reference channel for LFP data. My data is already referenced at rec level**

- [x] ** look into the grid mode in <field_index_map>. How does it determine different fields? Do we need to provide this info before hand? Is it determines multifields, what algo is used for this?**
NOTE: grid: fixed band tuned to typical grid spacing. *Make sure you change this to irregular spacing*

- [x] ** Add Max 50msec ts distance criteria to spike_ts and pos_ts matching** 

- [x] ** Make sure that spikes from cluster 0 are skipped.**
*Fixed: `load_ntt_spike_times` now unconditionally drops cell_number 0 (unsorted/noise), instead of only dropping it when other clusters were also present.*


*If hippocampus has both predctive and retrospective coding, why doesn't it have both theta phase precession and phase succession?*

## Theta Procession Fixes

- [ ] **You have multiple cells with twin peaks.Segregate spike wrt to theta phase peaks and re-plot phase precession. Go from peak of theta to minima for phase prec plot 1,all other spikes rerun for phase prec plot 2** 
*C:\Runita\NMR\analysis\AllSort_Results\PlaceCell\Data\PlaceCell_True\Fa1059\Day8\2_180\ThetaMod_PhasePrecession\TT4_SS_17.ntt probably phase precessing and processing.*

- [ ] **Have predefined fields. Inout his data into the code and calculate pass index for inividual fields seaparately and analyze for phase precession.**

- [ ] **Have specific ranges of theta phase for phase precessing and phase processing cells. The two have to occur at distinct theta phases.**

- [ ] **Procession is more common int ripple  based, reverse replay.**

- [ ] **Precession vs procession requires very small inter frame interval for tracking data (200 fps in Wang et.al. 2020). Caveats of 30 fps? Can we still get reliable procession?**

- [ ] **Low spike count can cause regression line to process. Compare spike numbers between precession and procession.**

- [ ] **Confounded by directional selecivity in 1D environemnts?** *Club this with directional selecetivity analysis. Check if a preffered direction of firing predicts precession or procession of the cell.*

- [ ] **TMI plot/polar plot should show either 1 single peak/cycle (unimodal cell) OR two peaks within a cycle, one out of which will be shorter than the other (bimodal cell).** *Quantify the peaks of these bimodal cells.* 
*This analysis is important because phase procession is more likely to occur during the short peaks of bimodal cells.*

- [ ] **Difference between procession and succession? Is a cell that's processing also necessarily precessing?**



### Pass index sawtooth bug:

<That saw-tooth "rise to 1, snap down, rise again" you're seeing is a real artifact — not the underlying phase-precession data — and it's a bug in this diagnostic script specifically, not in the main pipeline.
Root cause: pass_index_trace = angle(hilbert(...))/pi (line 108) wraps at ±1 — every time the true phase crosses an odd multiple of π (referenced to 0), the value snaps from +1 down to -1. That's expected and correct.
The bug is in segment_passes (line 117-126): it defines pass boundaries as
cycle = np.floor((unwrapped - unwrapped[0]) / (2 * np.pi))
i.e., every 2π of accumulated phase starting from wherever the trace happens to begin (unwrapped[0]). But the trace's actual wrap points (±1 snaps) occur at fixed offsets from 0, not from unwrapped[0]. Unless the recording happens to start exactly at a wrap point, every segment straddles one wrap discontinuity somewhere in its middle instead of at its edges — which is exactly the "starts at some fixed value (e.g. -0.75 for TT4_SS_17), rises to +1, snaps to -1, rises back to that same fixed value" pattern in your plots. The fixed start value differs per unit because unwrapped[0] differs per unit's trace, but it's constant across all passes within one unit — matching what you see.
For comparison, the real pipeline (compute_pass_index in ThetaMod_PhasePrecession.py:473) assigns each spike's pass index via np.mod(spk_unwrapped + np.pi, 2*np.pi) - np.pi, which re-wraps relative to 0 — so per-spike pass-index values themselves are fine. This is purely a mis-segmentation in this viz script's pass boundaries.
Fix: align segment boundaries with the same reference (multiples of π from 0) instead of unwrapped[0]:
cycle = np.floor((unwrapped + np.pi) / (2 * np.pi))>