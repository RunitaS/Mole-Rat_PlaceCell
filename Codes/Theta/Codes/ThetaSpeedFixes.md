# Theta Speed Fixes

- [ ] **Replace interpolation method with wavelet method.**

- [x] ** Add entire denoising paradigm to your code.**
Fixes:
ThetaVsSpeed.py:83-125 — added notch_filter/detrend_signal/lowpass_filter/_robust_high_outliers (matches ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean_v3.py).
ThetaVsSpeed.py:256-260 (extract_data) — the downsampled LFP is now line-noise notched (50/100/150/200 Hz), detrended, and low-passed at 100 Hz before theta-band filtering.
ThetaVsSpeed.py:380-396, 548-552 — reject_artifact_bins rejects BINSIZE (0.25 s) bins whose theta-signal peak-to-peak amplitude is a robust (MAD) outlier, before the bins are handed to drop_nan / the mixed model.

- [x] ** Instantaneous freq overshooting range bug:**
Fixes:
ThetaVsSpeed.py:223-235 — find_peaks_troughs now enforces a minimum spacing between detected peaks/troughs equal to one cycle at the bandpass's upper edge (7 Hz → ≥~143 ms apart at 1000 Hz). This is the actual root cause of frequencies exceeding 3–7 Hz: without it, a small secondary bump on an asymmetric theta cycle could get counted as its own peak, halving the apparent period and doubling the computed frequency (explaining the ~8–14 Hz cluster in your plot).
ThetaVsSpeed.py:430-431 — inst_freq/inst_power (computed on the 1000 Hz LFP timebase) were being binned with speed_ts (the 30 Hz tracking timebase) instead of lfp_ts, silently mispairing frequency/power values with the wrong time (and wrong speed). Now both use lfp_ts, matching how they were actually computed.