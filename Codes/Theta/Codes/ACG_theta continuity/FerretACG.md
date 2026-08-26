# ACG algo:

- How does it check for continuity?

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