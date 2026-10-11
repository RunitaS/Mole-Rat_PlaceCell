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


%Algorithm: Adaptive Binning Rate Map (Skaggs/Jim adaptive smoothing)
%This implements the classic adaptive smoothing method (Skaggs et al., 
% popularized in Jim Knierim's lab code) for computing spatial firing‑rate maps, 
% as opposed to fixed-kernel/Gaussian smoothing. The core idea: at each spatial bin, 
% grow a circular neighborhood radius until it contains "enough" occupancy/spike data 
% to give a statistically reliable rate estimate, then use that variable-size 
% neighborhood to estimate firing rate for that bin.

%Step by step (getAdaptiveBinnedRmap.m)
%Setup (lines 2-13)

%alpha is the smoothing parameter controlling how large the adaptive radius must grow 
% (scaled by frame rate since this occupancy map is in seconds, not raw sample counts, 
% to keep the exit criterion equivalent to the original pixel-count-based code).

%Output maps (abrmap, abroccMap) initialized to zero.
%Skip empty maps (line 14): only run if there's at least one spike anywhere.

%Per-pixel adaptive radius search (lines 15-44) — for every occupied bin (y,x):

%Compute dists, the Euclidean distance from every other bin to (y,x) via bwdist.
%Starting at radius r = 0, grow rsq (radius squared, incremented by 1 each iteration) and at each step:
%Sum spikes (Nspikes) and occupancy (NoccMap) within all bins where dists <= r.
%Test the exit criterion: alpha² · NoccMap² · rsq · Nspikes2 > 1. This is the Skaggs adaptive-smoothing 
% stopping rule — stop expanding once the disc contains enough combined occupancy and spike samples 
% relative to alpha (smaller alpha → larger radii → more smoothing).
%Loop capped at rsq < 200 (a practical bound from Jim's original code, not part of the formula).
%Rate assignment (lines 45-52):

%If the accumulated occupancy in the final disc is still below a minimum threshold (params(2), e.g. 0.5 s), 
% the bin is considered unreliable and set to 0.

%Otherwise, firing rate for that bin = Nspikes / NoccMap (spikes divided by time occupied within the adaptively-sized disc).
%Unvisited-bin coloring (lines 57-66): bins with zero final occupancy (never adequately sampled) are set to a 
% small negative sentinel value (cmin, a fraction of the peak rate) — likely used downstream to render unvisited 
% bins in a distinct color rather than as zero rate. Peak firing rate (peakFR_abrmap) is returned as the max of the raw rate map.

%Summary
%For each pixel, instead of using a fixed-size smoothing kernel, the algorithm expands a circular window 
% until it has collected enough occupancy+spike evidence (per the Skaggs criterion α²·N_occ²·r²·N_spikes > 1), 
% then computes rate = spikes/occupancy over that window. This gives finer spatial resolution where sampling 
% is dense (small radius) and coarser smoothing where sampling is sparse (larger radius) — an adaptive, 
% data-driven alternative to Gaussian smoothing.