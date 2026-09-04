# ACG algo:

ACG "Theta Continuity" Algorithm
This script implements the peak-range oscillation-quantification method from Dunn et al. 2022 (Nat. Commun.), which asks: for a short LFP snippet, how "clean" a theta oscillation is present, independent of its exact frequency? It does this by comparing the snippet's autocorrelogram (ACG) shape to a bank of pure-sinusoid ACGs. Below I trace the pipeline in actual execution order, from raw file to final plots.

1. Top-level driver ([ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean.py:1152](x:\NMR_group_data\Runita\Codes\myCodes\LFP\Tetrode\Sorted\ACG_theta continuity\ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean.py#L1152))
process_root_directory(ROOT_DIR) walks the directory tree, finds every .ncs file, and calls process_ncs_for_acg on each one, concatenating all per-epoch result rows into one big DataFrame. Metadata (animal/date/session/tetrode/channel) is parsed from the folder path via parse_metadata_from_path ([:305](x:\NMR_group_data\Runita\Codes\myCodes\LFP\Tetrode\Sorted\ACG_theta continuity\ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean.py#L305)) and attached to every row.

2. Loading one channel ([:133](x:\NMR_group_data\Runita\Codes\myCodes\LFP\Tetrode\Sorted\ACG_theta continuity\ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean.py#L133))
load_ncs memory-maps the Neuralynx .ncs binary format directly (skipping the 16 KB text header), reads 512-int16-sample records, concatenates them into one continuous trace, and converts raw ADC counts to microvolts via ADBitVolts. It also returns the UNIX timestamp of the very first sample (lfp_start_us), which is later used to align LFP epochs to the tracking data's clock.

3. Preprocessing the trace ([process_ncs_for_acg](x:\NMR_group_data\Runita\Codes\myCodes\LFP\Tetrode\Sorted\ACG_theta continuity\ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean.py#L641))
In order:

Resample 32000 Hz → 1000 Hz (FS_ACG) via signal.resample_poly. This matters because the original MATLAB code hardcodes its reference-sine time base at 1 kHz — resampling the data to the same rate keeps the reference bank's frequency axis meaningful in Hz.
Notch-filter 50/100/150/200 Hz mains harmonics with zero-phase IIR notches (filtfilt, Q=30).
Linear detrend to remove slow drift a notch filter won't touch.
4. Epoching and cleaning ([:565](x:\NMR_group_data\Runita\Codes\myCodes\LFP\Tetrode\Sorted\ACG_theta continuity\ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean.py#L565))
build_epochs chops the continuous trace into non-overlapping 1-second (1000-sample) windows, discarding any leftover tail.
reject_deltatheta_epochs: for each epoch, a Welch PSD is computed and delta-band (1–3 Hz) power is compared to theta-band (3–7 Hz) power. If delta ≥ theta, the epoch is likely dominated by slow artifact rather than a genuine oscillation, so it's flagged for rejection — this catches noise that a pure-amplitude threshold would miss (because it's not necessarily large-amplitude).
reject_artifact_epochs: computes each epoch's peak-to-peak amplitude, then flags amplitude outliers using a robust MAD-based z-score (_robust_high_outliers, threshold 5). Critically, the median/MAD reference statistics are computed only from epochs that already passed the delta/theta filter (ref_mask=keep_deltatheta), so already-bad epochs can't inflate the "typical" amplitude and mask other artifacts.
5. Movement labelling ([:186](x:\NMR_group_data\Runita\Codes\myCodes\LFP\Tetrode\Sorted\ACG_theta continuity\ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean.py#L186)–[:636](x:\NMR_group_data\Runita\Codes\myCodes\LFP\Tetrode\Sorted\ACG_theta continuity\ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean.py#L636))
find_position_file locates the one tracking .csv living alongside the .ncs file.
compute_velocity_from_position reads UNIX-us timestamps plus x/y position (cm). Before computing speed, _smooth_tracking_position cleans the trace:
Iteratively detects frame-to-frame jumps implying speed > 80 cm/s against the shrinking set of "good" samples (so runs of consecutive bad frames are caught, not just single spikes relative to the immediately preceding sample).
Bad frames are filled by linear interpolation between surviving good samples (not "hold last value", which would create an artificial speed spike when position snaps back).
The cleaned x/y traces are Gaussian-smoothed (sigma=1 sample).
_instantaneous_speed then computes frame-to-frame displacement ÷ actual elapsed time.
classify_epoch_movement: for each 1-second LFP epoch, using lfp_start_us to align to absolute UNIX time, the median tracking speed within that time window is computed, and the epoch is labelled 'moving' if median speed > 4 cm/s, 'immobile' if ≤ 4 cm/s, or 'other' (dropped) if speed couldn't be determined.
Only epochs that are both artifact-clean and labelled moving/immobile survive to the ACG stage (keep = keep_artifact & np.isin(label, ['moving','immobile'])).

6. The core ACG algorithm ([:442](x:\NMR_group_data\Runita\Codes\myCodes\LFP\Tetrode\Sorted\ACG_theta continuity\ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean.py#L442) quantify_xcorr_epochs, [:360](x:\NMR_group_data\Runita\Codes\myCodes\LFP\Tetrode\Sorted\ACG_theta continuity\ACG_theta_continuity_TT_Thresholded_EDmin_LFPclean.py#L360) create_sine_ref_xcorrs)
This is the mathematical heart of the script, translated from Dunn et al.'s MATLAB.

6a. Build the reference sinusoid bank (once per file, create_sine_ref_xcorrs)
For each candidate frequency f in 3–7 Hz stepped by 0.1 Hz (41 frequencies) <Should have had 46 frequencies, check why 5 freuqncies are dropped, also which 5 frequencies are dropped?>:

Synthesize a pure sine sin(2πf·t) sampled at 1 kHz, same length as one data epoch (1000 samples).
Compute its full autocorrelation (signal.correlate(..., mode='full')), giving 2N-1 lags, and normalise so the zero-lag value (the peak at the center) is 1. This produces a symmetric ACG shape unique to that frequency — a sine wave with a lower frequency has a "slower", more widely spaced ACG, and a higher frequency a "tighter" one.
Locate the first side peak: using find_peaks with a fixed prominence (0.05) on the ACG, find all local maxima; the zero-lag peak is at the center index; the "first side peak" is the very next maximum after center — this corresponds to one full oscillation cycle away from zero lag.
Locate the flanking troughs: the local minima immediately before and after that side peak.
refRange: this reference's own "peak range" = side-peak height − mean of the two trough heights. This is the sinusoid's reference peak-range value — note that for a pure sine, this quantity shrinks somewhat with increasing frequency (documented in the paper's Supp. Fig. 5h), simply because of how ACG shape and lag-window truncation interact — which is why the actual data's peak range gets divided by refRange later, to remove this frequency-dependent bias.
zero_crossings: find fractional (linearly interpolated) sample positions where the ACG crosses zero, and use these to define search windows — refP1range (bracketing the side peak), refT1range (bracketing the trough before it), refT2range (bracketing the trough after it). These windows are defined purely from the reference sinusoid's shape, and are what will later be applied to search the data's ACG for its analogous peak/troughs (since the data won't have an idealized shape to run find_peaks on reliably).
refED: the L2 (Euclidean) norm of each reference's full ACG vector — used later to normalise distances.
Output: for all 41 frequencies, a matrix of reference ACGs (refXC), their norms, their own peak ranges, and the three search-window index pairs.

6b. Score each data epoch against the bank (quantify_xcorr_epochs, per epoch)
For each kept 1-second epoch (a column of data_epochs):

Compute its full autocorrelation the same way as the references, normalised to zero-lag = 1. This gives the data epoch's own ACG vector, same length as the reference vectors.
Fit the best-matching frequency: compute the pointwise squared difference between the data ACG and every reference ACG, sum across lags (nansum, so NaN-padding from short epochs is ignored), square-root to get a Euclidean distance ED, then normalise each by that *reference's own norm* (<refED>) — giving <normED>. The reference frequency with the smallest normED is the epoch's frequency estimate (<freq_est>), and that minimum normalised distance is ED_min — a fit-quality metric: 0 means the data ACG looks essentially identical (in shape) to a pure sinusoid at that frequency; large ED_min means no sinusoid in the bank resembles the data's ACG shape well.
Fit-quality gate: if ED_min > ACG_ED_MIN_THRESH (default 1.0), the epoch is marked skipped=True and skipped_edmin=True — its peak-range measurement is considered untrustworthy since no sinusoid fit it well, but it's not dropped from the DataFrame (kept as a NaN/flagged row) so row alignment with epoch_index/file is preserved.
Measure the data's own peak/trough, using the matched reference's search windows (refP1range, refT1range, refT2range at the winning frequency index mi) rather than re-running peak detection on the (noisier) data ACG directly:
Peak: argmax of the data ACG within the reference's peak window.
Trough1 / Trough2: argmin within the reference's two trough windows.
Peak range = peak value − mean(trough1, trough2) (same formula as the reference).
Normalised peak range (<peakrangenorm>) = data's peak range ÷ that matched reference's own peak range (refRange[mi]). This is the key output metric: dividing by the frequency-matched sinusoid's peak range cancels out the frequency-dependent shrinkage noted in step 6a.5, so peakrangenorm close to 1 means the data's ACG has essentially the same peak-to-trough contrast as an ideal sinusoid at its estimated frequency (a clean, sustained oscillation), while close to 0 means the ACG is nearly flat there (no oscillatory structure, even though a "best-fit frequency" was still nominally assigned).
Output: a DataFrame with one row per kept epoch: ED_min, freq, peak1/trough1/trough2 (+ their lag indices), peakrange, peakrangenorm, skipped, skipped_edmin.

7. Back in process_ncs_for_acg
The per-epoch result DataFrame gets movement, median_speed_cms, epoch_index (original epoch position, for later waveform re-derivation), and the absolute file path attached, then is returned up to process_root_directory.

8. Aggregation and plotting
summarize_peak_range: median/IQR/n of peakrangenorm per (animal, channel, movement) group, excluding skipped rows.
plot_moving_vs_immobile: split violin plots of normalised peak range, moving vs. immobile, per channel — reproduces the paper's depth-profile figures.
compute_group_centroids / pick_example_epoch: for the Figure-4-style panels, computes mean (freq, peakrangenorm) per movement group, and picks the single epoch closest to each group's centroid as a representative example.
plot_example_epoch: draws one epoch's raw trace plus its ACG with the matched reference sinusoid overlaid and the peak-range measurement highlighted in green — this directly visualizes steps 6a–6b for one epoch.
_draw_freq_peakrange_scatter / plot_freq_peakrange_summary / plot_freq_peakrange_by_movement / plot_figure4_style: assemble the frequency-vs-normalised-peak-range scatter (colored by running speed for moving epochs, black/outlined for immobile), with marginal histograms and centroid diamonds — reproducing Dunn et al. Fig. 4e/h, plus an annotation reporting what fraction of epochs were excluded by the ED_min gate.

## Summary
For every clean 1-second LFP epoch: fit the best-matching pure-sinusoid frequency by ACG shape (3–7 Hz search), then measure how "sinusoid-like" (peaked vs. flat) the epoch's own ACG is around that frequency's characteristic lag, normalized so the metric is frequency-independent. peakrangenorm near 1 = strong, continuous theta-like rhythm in that second of LFP; near 0 = no real oscillation despite a nominal frequency estimate. Comparing this metric's distribution between moving and immobile epochs (per channel/animal) is exactly the "theta continuity during immobility" test from the paper's Figure 4.




## ED_min calculation

**How ED_min is computed**
ED_min is produced inside quantify_xcorr_epochs, for each epoch, in three stages:

1. Build a reference bank of "pure oscillation" autocorrelograms (create_sine_ref_xcorrs)
For each candidate frequency f in the reference range (ACG_FREQ_RANGE, stepped by ACG_FREQ_RES), a pure sine wave is synthesized and self-correlated:


refsig = np.sin(2*np.pi*freq*reft)
refxc  = correlate(refsig, refsig, 'full') / max(...)   # peak-normalised to 1 at zero-lag
Each reference column refXC[:, n] is a normalized autocorrelogram — 1 at zero lag, oscillating between roughly −1 and 1 elsewhere, depending on frequency. refED[n] = ||refXC[:, n]||₂ is the Euclidean (L2) norm of that reference vector across all lags (L431).

2. Compute the data epoch's own normalized autocorrelogram (L493-L494)


xc = correlate(dataepoch, dataepoch, 'full') / max(...)   # also peak-normalised to 1 at zero-lag
3. Compare the data's ACG shape against every reference, keep the best match (L498-L505)


sq_diff = (refXC - xc[:, None])**2
ED      = sqrt(nansum(sq_diff, axis=0))   # Euclidean distance to each reference, lag-by-lag
normED  = ED / refED                       # normalised by that reference's own norm
mi      = argmin(normED)
ED_min[n]  = normED[mi]                    # distance to the *closest*-matching reference
freq_est[n] = ref_freqs[mi]                 # the frequency of that best-matching reference
So ED_min is the normalized Euclidean distance between the epoch's actual autocorrelogram shape and the single best-fitting pure-sinusoid autocorrelogram from the reference bank, at whatever frequency minimizes that distance. 0 = the data's ACG is (nearly) identical to some pure sinusoid's ACG — clean, strongly rhythmic activity. Larger values mean the epoch's ACG shape doesn't resemble a pure oscillation at any tested frequency.

What's the maximum value it can take?
There's no fixed, built-in ceiling (like the [−1, 1] bound of a correlation coefficient). It's a ratio of Euclidean distance to the reference's own norm, so its scale is geometry-dependent, not a normalized similarity score. That's exactly why L114-L116 says "There's no universal cutoff — inspect the ED_min column's distribution in your own data... before picking a value."

That said, you can reason about the practical range:

Both xc and each refXC column are pinned to 1 at zero lag (autocorrelation always peaks there) and otherwise bounded in roughly [-1, 1].
If the data has essentially no oscillatory structure (e.g., noise-like, delta-shaped ACG), ED_min tends toward a value just under 1 (worked out from the math: sqrt(1 - 1/refED²), which → 1 as the reference gets long/oscillatory).
If the data's ACG is actively out-of-phase / dissimilar in shape from every candidate reference (e.g., an oscillation frequency well outside the reference bank, or a noisy/irregular ACG that anti-correlates with the sinusoid shape over much of the lag range), the distance can exceed 1. A rough geometric estimate (treating a sinusoid's mean-square power as ~0.5) puts the plausible practical ceiling around ~2, though this isn't a hard proven bound — it depends on epoch length, the reference frequency range, and NaN padding.
Your observed range of 0.0–1.6 is consistent with this — it's within the plausible envelope and not a sign of a bug. There isn't a clean textbook maximum (no "max is 1" or "max is 2" guarantee) — the code's own guidance is correct: plot results_df['ED_min'].hist() on your data and pick ACG_ED_MIN_THRESH based on where the distribution separates "good sinusoidal fits" from "poor fits," rather than assuming a universal cutoff.


# Soraya Dunn MATLAB code:

function [XC, xcTbl] = quantify_xcorr_epochs(data_epochs,freq_range,freq_resolution)
try
nepochs   = size(data_epochs,2);
winlength = size(data_epochs,1);

% preallocate
XC        = NaN(winlength*2-1,nepochs);
mindist   = NaN(nepochs,1);
mindisti  = NaN(nepochs,1);
xcvals    = NaN(3,nepochs);
xcIdx     = NaN(3,nepochs);
peak1rangeED     = NaN(nepochs,1);
normpeak1rangeED = NaN(nepochs,1);
skipped = false(nepochs,1);
% create referene sine xcorr bank
ref_freqs = freq_range(1):freq_resolution:freq_range(2);
[refXC, refED, refRange,refP1range, refT1range, refT2range] = create_sine_ref_xcorrs(ref_freqs,size(data_epochs,1));

for n = 1:nepochs
    
    dataepoch = data_epochs(~isnan(data_epochs(:,n)),n);
    
    if numel(dataepoch)==1 % skip if only 1 data point
        mindisti(n) = 1; % so doesn't throw error below
        skipped(n)  = true;
        continue
    end
    
    xc = xcorr(dataepoch,dataepoch); % calc autocorrelation
    xc = xc ./ max(xc); 
    XC(1:length(xc),n) = xc;
     
    %% eucdist method 
     % find matching sine with min ED
    XC1 = repmat(XC(:,n),1,size(refXC,2));
    ED  = naneucdist(XC1',refXC');
    normED = ED./refED';
    [md,mi]= min(normED); 

    mindist(n)  = md;
    mindisti(n) = mi;
    
    % find peak range of data autocorr
    % find value at peak max
    [peakmax,peakmaxi] = max(XC1(refP1range(1,mi):refP1range(2,mi)));
    peakmaxi = peakmaxi + refP1range(1,mi)-1;
    % find min in trough before peak
    [trough1min,trough1mini] = min(XC1(refT1range(1,mi):refT1range(2,mi)));
    trough1mini = trough1mini + refT1range(1,mi)-1;   
    % find min in trough after peak
    [trough2min,trough2mini] = min(XC1(refT2range(1,mi):refT2range(2,mi)));
    trough2mini = trough2mini + refT2range(1,mi)-1;   
    
    xcvals(1,n) = peakmax;
    xcvals(2,n) = trough1min;
    xcvals(3,n) = trough2min;
    
    xcIdx(1,n) =  peakmaxi;
    xcIdx(2,n) =  trough1mini;
    xcIdx(3,n) =  trough2mini;
    
    peak1rangeED(n) = peakmax - mean([trough1min,trough2min]);
    normpeak1rangeED(n) = peak1rangeED(n)/refRange(mi);   % normalise by peakrange of reference sine

    
end


% table output
xcTbl = table;

xcTbl.EDmin         = mindist;
xcTbl.freq          = ref_freqs(mindisti)';
xcTbl.freq(skipped) = NaN;
xcTbl.peak1         = xcvals(1,:)';
xcTbl.peak1i        = xcIdx(1,:)';
xcTbl.trough1       = xcvals(2,:)';
xcTbl.trough1i      = xcIdx(2,:)';
xcTbl.trough2       = xcvals(3,:)';
xcTbl.trough2i      = xcIdx(3,:)';
xcTbl.peakrange     = peak1rangeED;
xcTbl.peakrangenorm = normpeak1rangeED;


catch err
    parseError(err)
    keyboard
end


end



function [refXC, refED, refRange, refP1range, refT1range, refT2range] = create_sine_ref_xcorrs(ref_freqs,datsize)

reft = 0:1/1000:(datsize-1)/1000;

refXC      = NaN(length(reft)*2-1,length(ref_freqs)); % preallocate
refRange   = NaN(numel(ref_freqs),1);
refPTIdx   = NaN(3,numel(ref_freqs));
refP1range = NaN(2,numel(ref_freqs));
refT1range = NaN(2,numel(ref_freqs));
refT2range = NaN(2,numel(ref_freqs));

for n = 1:numel(ref_freqs) % for each frequency
    refsig = sin(2*pi*ref_freqs(n)*reft);  % calc sine
    refxc = xcorr(refsig,refsig);          % autocorrelogram of sine
    refxc = refxc./max(refxc);
    refXC(:,n) = refxc;
    
    %% find peak range for reference sine autocorrelogram
    [maxP,minP] = findMinMax(refxc,0.05,'fixed');   % find extrema

    midpeaki = find(maxP(:,1)==datsize);  % find peak/troughs of interest (first peak after centre)
    peak1 = maxP(midpeaki+1,:);
    trough1i = find(minP(:,1)<peak1(1));
    trough1i = trough1i(end);
    trough2i = find(minP(:,1)>peak1(1));
    trough2i = trough2i(1);
    trough1 = minP(trough1i,:);
    trough2 = minP(trough2i,:);
    
    refRange(n) = peak1(2) - mean([trough1(2),trough2(2)]); % find peak range for sine autocorrelogram
    
    refPTIdx(1,n) = trough1(1);
    refPTIdx(2,n) = peak1(1);
    refPTIdx(3,n) = trough2(1);   
    
    %% find regions over which max/min of data autocorr will be found
    interceptpoints = zero_crossings(refxc);
    belowt1 =interceptpoints(interceptpoints < trough1(1));
    abovet2 =interceptpoints(interceptpoints > trough2(1));
    abovep1 =interceptpoints(interceptpoints > peak1(1));
    belowp1 =interceptpoints(interceptpoints < peak1(1));
    
    refP1range(1,n) = round(belowp1(end));
    refP1range(2,n) = round(abovep1(1));
    refT1range(1,n) = round(belowt1(end));
    refT1range(2,n) = round(belowp1(end));
    refT2range(1,n) = round(abovep1(1));
    refT2range(2,n) = round(abovet2(1));
    
end
refED = vecnorm(refXC); % calc euc. dist of each ref sine AC
end