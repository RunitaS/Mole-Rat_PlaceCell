# Traingular Kernel Vs Adaptive Binning

Your Python approach (fixed-bin + triangular smoothing):

Space is divided into a fixed grid of target_bin_cm (2 cm) bins, identical everywhere in the arena.
Firing rate per bin = spike count / occupancy time, computed independently per bin.
A 3×3 triangular kernel is then convolved over the whole rate map to smooth it, with edge-correction by dividing by the convolved occupancy mask (so poorly-sampled edge bins don't get diluted by phantom zero-neighbors).
The smoothing radius is fixed (one bin ring) regardless of how much occupancy data exists in that bin — a bin visited for 50 s and one visited for 1.1 s (your min_occ_s floor) get the same smoothing footprint.
Adaptive binning (the classic method getAdaptiveBinnedRmap is almost certainly implementing):

Instead of smoothing a fixed grid, each pixel/bin grows a circular neighborhood radius outward until the accumulated occupancy and/or spike count within that circle satisfies a criterion (commonly radius grows until occupancy > α / (spikes × occupancy^0.5) or similar, per Skaggs et al.).
The firing rate at that point = spikes-in-circle / time-in-circle, using however large a circle was needed to get a reliable estimate.
This means the effective smoothing kernel size varies spatially: sparsely-sampled or low-firing regions get a large averaging radius (more smoothing, more bias, but stable estimate), while densely-sampled/high-firing regions keep a small radius (less smoothing, more spatial precision).
Key practical differences:

Fixed vs. variable kernel — your triangular smoothing always looks at the same 3×3 (or effectively ~1-bin-radius) neighborhood; adaptive binning expands/contracts per-location based on sampling density.
Bias-variance tradeoff handling — adaptive binning is specifically designed to suppress noise in undersampled bins (e.g., near arena edges, or in maze arms visited briefly) without over-smoothing well-sampled central regions. Fixed smoothing either under-smooths sparse bins (noisy peak_fr estimates) or over-smooths dense ones, depending on kernel size chosen.
min_occ_s thresholding — your pipeline handles undersampling by hard-excluding bins below 1 s occupancy (valid_mask) rather than adaptively growing the estimation window there. Adaptive binning instead keeps those locations but computes their rate from a larger effective area.
Peak firing rate / SIR bias — because adaptive binning can produce very large radii in sparse regions, it tends to be more conservative (lower apparent peak rates in undersampled zones) but is considered the more statistically defensible choice when occupancy is highly non-uniform (e.g., open-field foraging with corners rarely visited). Fixed triangular smoothing is simpler and computationally cheap but can let small-N bins produce spuriously high rates if min_occ_s isn't strict enough, or oversmooth true fields if the kernel is too wide relative to place field size.
If you want, share the actual getAdaptiveBinnedRmap.m file path (or paste it) and I'll give you a line-by-line comparison against _triangular_smooth/compute_metrics instead of the generic textbook description.

c#1-69
function [abrmap,peakFR_abrmap]=getAdaptiveBinnedRmap(occMap,spikeMap,params)
frameRate=params(15);
%%                                  Sachin's Adaptive binning
%changes between sachin's and my code- IJ's occmap has already been converted
%to seconds while his seems to be in pixels. Hence, alpha has been
%multiplied by framerate to account for this in the exit criteria
alpha = 0.0001 *frameRate; %Jim uses alpha of 0.0001. Skaggs c code uses alpha of 0.001.
                                          % IJ's occmap is in seconds while that of Sachin's was in counts.
                                          % So, scaling alpha with the same number to make sure the exit
                                          % criteria remains consistent
z = zeros(size(occMap));
abrmap = z;
abroccMap = z;
if max(max(spikeMap))>0 %make sure that there is atleast 1 spike in the spike map.
    for x = 1:size(occMap,2)
        for y = 1:size(occMap,1)
            if occMap(y,x) > 0
                rsq = 0;
                Nspikes2 = 1; %pretend there's atleast 1 spike, and 1 pixel occupied in occMap. This is needed to
                              %avoid 0 threshold. This gets added to each pixel in Jim's and Skaggs' calculations. 
                              %Sachin changed code to accomodate this offset : sac 1/7/09. changed again to use it
                              %only in the Enoughpoints Threshold calculation, not in actual ratemap: sac 1/19/09
                NoccMap = occMap(y,x);
                d = z;
                d(y,x) = 1;
                dists = bwdist(d);
                EnoughPoints = 0;
                while (rsq < 200) && (EnoughPoints == 0) %rsq<200 condition comes from Jim's program, not the formula. 
                                                         %rsq here is equivalent to r in Jim's code - Buildtable defines 
                                                         %r = i*i + j*j : square of the hypotaneous of a right angled triangle.
                    r = sqrt(rsq);
                    foo = spikeMap(dists <= r);
                    Nspikes = sum(foo); %Jim's code (and Skaggs') has offset of 1 spike and 1 occMap. Not a requirement of the formula.
                    bar = occMap(dists <= r);
                    
                    NoccMap = sum(bar);
                    if Nspikes > 0
                        Nspikes2 = Nspikes;
                    end
                    if alpha*alpha*NoccMap*NoccMap*rsq*Nspikes2 > 1
                        EnoughPoints = 1;
                    end
                    rsq = rsq + 1;
                end
                if  NoccMap< params(2) % IJ's occmap is already in secs and  < 0.5sec occupancy. both Skaggs and Jim use this cut off. though I think occpancy for this pixel needs to be set to zero, too. (can't see it done in Jim's code). sac 1/7/09. fixed occMapupancy issue: sac 1/19/09
                    abrmap(y,x) = 0;   
                    abroccMap(y,x) = 0;
                else
                    abrmap(y,x) = Nspikes/NoccMap;  %IJ's occmap is already in sec, unlike sachin's
                    %abrmap(y,x) = frameRate* Nspikes/NoccMap;
                    abroccMap(y,x) = NoccMap;
                end
            end
        end
    end
end
abr_max = max(max(abrmap));
if abr_max > 0
    cmin = -(abr_max/255);
else
    cmin = -1;
end

abrmap(abroccMap==0) = cmin; 
%meanFR_abrmap=spikeNum/sum(sum(occMap));
peakFR_abrmap=abr_max ;


return
Now I have the actual MATLAB implementation. Here's the precise comparison.

Core mechanism
MATLAB (getAdaptiveBinnedRmap.m) — classic Skaggs/Muller-Kubie adaptive smoothing, per pixel:

For each occupied pixel (y,x), grow a circular disc of radius r = sqrt(rsq) outward from that pixel (getAdaptiveBinnedRmap.m:24-38), summing spikeMap and occMap inside the disc as it grows.
The disc stops growing once alpha² · NoccMap² · rsq · Nspikes2 > 1 (getAdaptiveBinnedRmap.m:33) — i.e., once enough combined spike count and occupancy time have accumulated to trust the estimate, or rsq hits the hard cap of 200 (r ≈ 14.1 px).
The rate at that pixel is then Nspikes/NoccMap computed directly from the raw sums over that one circle — there's no separate "unsmoothed map" step; smoothing IS the rate estimate.
Radius is different for every pixel — sparse/low-firing regions get a big disc (heavy averaging, high confidence), well-sampled/high-firing regions keep a small disc (near-raw resolution).
Occupancy cutoff (params(2), typically 0.5 s) is applied after the disc has grown, to the accumulated NoccMap over the final circle (getAdaptiveBinnedRmap.m:41).
Python (_triangular_smooth in Fig1b_PlaceCellCharacterization.py:151-179) — fixed-kernel convolution:

fr_raw = spike_map / occ_map is computed per-bin first, independently, unsmoothed.
A single fixed 3×3 triangular kernel (weights 1-2-4-2-1, i.e. effectively a 1-bin radius, ~2 cm) is convolved over fr_raw and separately over the binary valid_mask, then divided — every bin gets exactly the same-size neighborhood regardless of how much data it has.
Occupancy cutoff (min_occ_s = 1.0 s) is applied before any smoothing, per raw bin (Fig1b_PlaceCellCharacterization.py:390), hard-excluding under-visited bins rather than letting them borrow support from a wider area.
The consequential differences
Fixed vs. variable footprint — this is the fundamental difference. MATLAB's disc radius is chosen per pixel to satisfy a statistical reliability criterion; Python's kernel radius (1 bin) is the same everywhere, with no mechanism to widen when a bin is poorly sampled beyond the binary yes/no min_occ_s gate.

Weighting basis of the smoothing itself — MATLAB sums raw spike counts and raw occupancy time over the disc, so the final ratio is naturally weighted by how long each contributing pixel was actually occupied. Python's convolution averages already-computed per-bin rates (fr_raw), weighted only by the triangular kernel coefficients and by whether a neighbor bin is "valid" (a 0/1 mask) — not by how much occupancy time that neighbor actually has. A neighbor visited for 1.01 s counts the same as one visited for 20 s.

Where the occupancy floor bites — MATLAB applies its floor to the occupancy accumulated over the final, possibly large disc; Python applies it to the raw single bin before smoothing ever happens. This means Python can throw away a bin outright even though its neighbors could have supported a reliable estimate, whereas MATLAB's growing disc would have simply expanded to pick up that support.

Practical consequence — MATLAB's adaptive method produces more statistically stable peak/place-field estimates in sparsely-sampled regions (arena corners, maze arms visited briefly) at the cost of blurring detail there, while your Python pipeline's fixed 3×3 kernel is cheap and vectorized but can leave residual noise in bins just above the 1 s cutoff, or under/over-smooth uniformly regardless of local sample size.

If you wanted the Python pipeline to actually match getAdaptiveBinnedRmap.m's behavior (not just approximate it), it would need a genuine per-bin growing-disc rewrite (distance-transform-based, vectorizable but not a simple convolution swap) — happy to help build that as a drop-in replacement for _triangular_smooth if that's useful.