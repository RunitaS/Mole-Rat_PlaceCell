function [ret,beta,f0]=speed_firing_runita(posx,posy,posts,ts,PosSampleRate)
% ------------------------------------------------------------------------
% Relates a neuron's firing rate to the animal's running speed.
% INPUTS:
%   posx, posy    : column vectors of x / y position (one row per position sample)
%   posts         : position timestamps (in ms, judging by the 1000/rate step below)
%   ts            : spike (action-potential) timestamps, same time base as posts
%   PosSampleRate : position sampling rate in Hz (e.g. 25)
% OUTPUTS:
%   ret  : struct with correlation/regression results (r, p_value, slope, intercept)
%   beta : slope of the speed-vs-firing-rate line
%   f0   : y-intercept of that line (firing rate extrapolated to speed 0)
% ------------------------------------------------------------------------

speed = [];                                   % initialise an empty vector to hold per-sample distances
for o = 1:length(posx)-1                       % loop over every position sample except the last (need pairs)
    speed(o) = sqrt(((posx(o+1,1)-posx(o,1)).^2)+((posy(o+1,1)-posy(o,1)).^2));
    % Euclidean distance travelled between sample o and o+1:
    %   sqrt( dx^2 + dy^2 ), i.e. straight-line distance in position units (cm)
end
% PosSampleRate = 25;                          % (commented out) hard-coded rate; now passed in as an argument

speed = speed*PosSampleRate;                   % distance-per-sample * samples-per-second = distance-per-second (speed, cm/s)
speed = speed';                                % transpose from a row vector to a column vector
speed(end+1,1)=speed(end,1);                   % pad by repeating the last value so speed length == number of position samples
                                               % (the loop produced N-1 values for N samples)

InstSpeed = speed;                             % rename: this is the "instantaneous speed" trace, one value per position sample

Alltime = [posts(1):1000/PosSampleRate:posts(end)];
% Build a vector of time-bin edges spanning the whole recording.
% Step = 1000/PosSampleRate ms  ->  one bin per position sample (implies posts is in milliseconds)

[FirRate,x]= histc(ts,Alltime);                % count how many spikes (ts) fall into each time bin -> spike count per bin
                                               % x = index telling which bin each spike landed in (unused here)
FirRate = FirRate * PosSampleRate;             % convert spike COUNT per bin to a RATE in Hz (count / bin-width, bin-width = 1/rate s)

Speedsmoothingwindows = 0.08;                  % smoothing window length in seconds (0.08 s = 80 ms); note: name says "speed" but it smooths firing rate
G = fspecial('gaussian',[Speedsmoothingwindows/PosSampleRate^-1,1],4);
% Build a Gaussian smoothing kernel:
%   PosSampleRate^-1 = 1/PosSampleRate (seconds per sample)
%   Speedsmoothingwindows / (1/PosSampleRate) = 0.08 * PosSampleRate = window length in SAMPLES
%   [.. , 1] makes it a column kernel; the 4 is the Gaussian standard deviation (sigma)

InstFirRate = filtfilt(G,1,FirRate);           % zero-phase (forward+backward) filtering of the rate with the Gaussian -> smoothed rate, no time shift
InstFirRate(numel(ts)+1:end) = [];             % trim the smoothed rate down to as many elements as there are spikes
                                               % (unusual step: it discards any rate samples beyond count(ts))

minspeed = 4; %(cm/s)                           % lowest speed to include (below this = likely non-locomotor/rest, excluded)
maxspeed = 100;                                % highest speed to include (above this = rare/noisy, excluded)
speedBinLengthCm = [2];                        % width of each speed bin, in cm/s
speedBins = [minspeed:speedBinLengthCm:maxspeed+speedBinLengthCm];
% Speed-bin EDGES from 4 to 102 in steps of 2 (the +binLength gives an extra edge past maxspeed)

fit2bins = [1,numel(find(speedBins<maxspeed+speedBinLengthCm))];
% Which bins to include in the regression fit: from bin 1 up to the count of edges below (maxspeed+binLength)
% -> effectively "first bin" to "last valid bin" (updated again later)

ToCompare = find(InstSpeed>minspeed & InstSpeed<maxspeed);
% Indices of the time samples whose speed lies strictly within [minspeed, maxspeed]

ToCompare(find(ToCompare > numel(InstFirRate)))=[];
% Drop any of those indices that fall beyond the (trimmed) length of InstFirRate, to keep indexing safe

RawPearsonCorrelation = nancorr(InstSpeed(ToCompare),InstFirRate(ToCompare));
% Pearson correlation between speed and firing rate over the valid samples, ignoring NaNs
% (nancorr is a custom/external helper, not built-in MATLAB)

if isnan(RawPearsonCorrelation)                % if correlation could not be computed (e.g. no valid data / constant signal)...
    ret.r = NaN;                               %   ...return NaN for the correlation coefficient
    ret.p_value = NaN;                         %   ...NaN for the p-value
    ret.slope = NaN;                           %   ...NaN for the slope
    ret.intercept = NaN;                       %   ...NaN for the intercept
%     ret = [];                                %   (alternative, commented out: return empty struct instead)
    beta = [];                                 %   ...empty slope output
    f0 = [];                                   %   ...empty intercept output
else                                           % otherwise, we have usable data -> do the full binned analysis
%%
speed2 = InstSpeed(ToCompare);                 % speed values for the valid samples only
instFreq2 = InstFirRate(ToCompare);            % matching firing-rate values for the valid samples
nBins=length(speedBins)-1;                     % number of speed bins = number of edges minus 1

[nPerBin, binInd]=histc(speed2, speedBins);    % nPerBin: how many speed samples fall in each bin
                                               % binInd : which bin index each sample in speed2 belongs to
nPerBin         =nPerBin(1:end-1);             % drop the final histc entry (samples exactly on the last edge / out of range)
dataPtsPerBin=nPerBin(1:end-1);                % copy but drop one more end value (kept separately for the "% data per bin" calc below)

meanFreqPerBin=zeros(nBins,1);                 % pre-allocate: mean firing rate in each bin
stdFreqPerBin=zeros(nBins,1);                  % pre-allocate: std of firing rate in each bin
binCentres=zeros(nBins,1);                     % pre-allocate: centre speed of each bin
binSpeed = zeros(nBins,1);                     % pre-allocate: mean actual speed of samples in each bin
semFreqPerBin = zeros(nBins,1);                % pre-allocate: standard error of the mean firing rate per bin
for nnBin=1:nBins                              % loop over every speed bin
    meanFreqPerBin(nnBin)=nanmean(instFreq2(binInd==nnBin)); % mean firing rate of samples that fell in this bin
    stdFreqPerBin(nnBin)=nanstd(instFreq2(binInd==nnBin));   % standard deviation of those firing rates
    binCentres(nnBin)=nanmean([speedBins(nnBin); speedBins(nnBin+1)]); % centre = average of this bin's two edges

    binSpeed(nnBin)=nanmean(speed2(binInd==nnBin)); % mean of the actual speeds within this bin
    semFreqPerBin(nnBin) = stdFreqPerBin(nnBin)/sqrt(nPerBin(nnBin)); % SEM = std / sqrt(n) for this bin
end
MaxSpeedBin = speedBins(end-1);                % the second-to-last edge, used as an upper cutoff on bin centres
MaxNumberOfBins = numel(find(binCentres<MaxSpeedBin)); % how many bin centres lie below that cutoff
meanFreqPerBin(MaxNumberOfBins+1:end)=[];      % discard bins beyond the cutoff (mean rate)
stdFreqPerBin(MaxNumberOfBins+1:end)=[];       % discard bins beyond the cutoff (std)
binCentres(MaxNumberOfBins+1:end)  =[];        % discard bins beyond the cutoff (centres)
semFreqPerBin(MaxNumberOfBins+1:end)  =[];     % discard bins beyond the cutoff (SEM)

fit2bins(2) =MaxNumberOfBins;                  % update the fit range's upper bound to the last kept bin

%errorbar(binCentres,meanFreqPerBin,semFreqPerBin)  % (commented) quick plot of rate vs speed with error bars
%%% FIT REGRESSION LINE TO VELOCITY VS EEG FREQ PLOT %%%%%%%%%%%%%%%%%%%%%
% Aim to deterimin slope of line and y-axis intersect (under oscillatory
% model these are called Beta and F0 respectivly).
% NB only use a subset of points to make this fit - very low speeds will be
% contamintated with non-movement related theta and high speeds are very
% under sampled.
% (These comments are inherited from an EEG-theta-frequency version of the analysis.)


%Use regress to do least squares best fit - seems that regress can also
%give the y-axis crossing (c) if the first set of y values it is gives is a
%column of ones, second value returned is Beta where y= Beta.x + c
%hence lsCoeff is [2 x 1] where lsCoeff(1) is c
switch lower('lsBinned')                       % NOTE: switched on a hard-coded string -> only the 'lsbinned' case ever runs
    case 'lsbinned'                            % least-squares fit to the BINNED means
        lsCoeff = regress(meanFreqPerBin(fit2bins(1):fit2bins(2)), ...
            [ones(fit2bins(2)-fit2bins(1)+1,1) binCentres(fit2bins(1):fit2bins(2))]);
        % regress(y, [ones, x]) -> lsCoeff(1)=intercept, lsCoeff(2)=slope
        % (this result is not actually used later; see CorrelateXYAndRegression below)

    case 'lsraw'                               % (dead code) least-squares fit to the RAW unbinned samples
        %Decide which points are in range
        lowerSpeedLimit=speedBins(fit2bins(1));            % lower speed edge of the fit range
        upperSpeedLimit=speedBins(fit2bins(2)+1);          % upper speed edge of the fit range
        validSpeed=speed>=lowerSpeedLimit & speed<upperSpeedLimit; % logical mask of in-range raw samples
        lsCoeff = regress(instFreq(validSpeed), [ones(sum(validSpeed),1), speed(validSpeed)]); % fit rate vs speed

    case 'robustraw'                           % (dead code) robust (outlier-resistant) fit to the raw samples
        %Decide which points are in range
        lowerSpeedLimit=speedBins(fit2bins(1));            % lower speed edge
        upperSpeedLimit=speedBins(fit2bins(2)+1);          % upper speed edge
        validSpeed=speed>=lowerSpeedLimit & speed<upperSpeedLimit; % in-range mask
        lsCoeff=robustfit(speed(validSpeed),instFreq(validSpeed));  % robustfit returns [intercept; slope]
end
PercentageDataPtsPerBin = dataPtsPerBin/sum(dataPtsPerBin)  ; % fraction of all data points contributed by each bin
meanFreqPerBinToRegress = meanFreqPerBin;      % copy the per-bin means; this copy will be cleaned before fitting
meanFreqPerBinToRegress(find(PercentageDataPtsPerBin < (1/100) ) )    = NaN ;
% Blank out (set NaN) any bin holding less than 1% of the data, so sparse/unreliable bins don't bias the fit

in.x = binCentres;                             % x data for the correlation/regression helper = bin centre speeds
in.y =meanFreqPerBinToRegress;                 % y data = cleaned mean firing rate per bin
in.PLOT_ON = 1;                                % tell the helper to draw the plot
in.xlabel = 'cm/s';                            % x-axis label (speed)
in.ylabel = 'Rate (Hz)';                       % y-axis label (firing rate)
in.title = 'Speed correlation';                % plot title

[ ret ] = CorrelateXYAndRegression(in);        % custom helper: computes r, p-value, slope, intercept and plots the fit
beta = ret.slope;                              % slope of the speed-vs-rate line (rate gain per cm/s)
f0   = ret.intercept;                          % intercept: rate extrapolated to 0 cm/s
% beta=lsCoeff(2);                             % (commented) alternative: take slope from the regress() result above
% f0=lsCoeff(1);                               % (commented) alternative: take intercept from the regress() result above

%And for comparsion use the best fit line to estimate theta freq at
%15cm/s
freqAt15cm = (15*beta) + f0;                   % predicted firing rate (or theta freq) at a reference speed of 15 cm/s
                                               % (computed but not returned as an output)

end                                            % end of the if/else (valid-data) block

%Not using it at the moment but the matlab function robustfit will also do
%linear fitting but using somethign more sophisticated than least squares
%(ls is suseptable to outliers). Again returns a [2 x 1]
%NB doesn't seem to work too well
% rbCoeff = robustfit(trial(nEEG).meanFreq(lowerBin:upperBin), ...
%     trial(nEEG).binMidPt(lowerBin:upperBin));

end                                            % end of the function

%%%
% 
% This is a neuroscience analysis function that measures how a neuron's firing rate depends on the animal's running speed. You feed it the animal's tracked position over time (posx, posy, posts), the times the neuron fired (ts), and the position sampling rate. It:
% 
% Computes instantaneous running speed from consecutive position samples.
% Builds an instantaneous firing rate by histogramming spike times into position-rate time bins and smoothing.
% Computes a raw Pearson correlation between speed and firing rate.
% Bins the data by speed, averages the firing rate in each speed bin, and fits a regression line to get a slope (beta) and intercept (f0).
% 
% The beta/f0 naming and several comments come from the oscillatory-interference model of theta rhythm, where EEG theta frequency ≈ f0 + beta·speed. Here that same machinery is being reused on firing rate instead of EEG frequency.
% 
% A few things worth flagging up front, which I've marked in the code:
% 
% The switch lower('lsBinned') is switched on a hard-coded string, so only the 'lsbinned' branch ever runs — the 'lsraw' and 'robustraw' branches are dead code, as is the lsCoeff result (the real beta/f0 come from CorrelateXYAndRegression).
% nancorr and CorrelateXYAndRegression are custom/external helper functions, not built-in MATLAB.
% The line InstFirRate(numel(ts)+1:end) = [] trims the firing-rate vector to the number of spikes, which is an unusual choice — I've commented what it does mechanically rather than assuming intent.
% There are some off-by-one/redundant trimming steps around the bin counts; I've described each mechanically.
% 
% Here's the fully commented version:
% 
% Created a file, read a file
% Created a file, read a file
% 
% The commented file is ready above. To recap the flow so the pieces hang together:
% 
% Speed comes from the frame-to-frame Euclidean distance in position, scaled by the sample rate to give cm/s. Firing rate comes from histogramming spike times into per-frame time bins, scaling to Hz, then Gaussian-smoothing with filtfilt (zero phase lag). After restricting to samples in the 4–100 cm/s window, it takes a raw Pearson correlation, and if that's valid it proceeds to bin the rate by speed, average within each 2 cm/s bin, drop under-sampled bins (<1% of data), and fit a line whose slope is beta and intercept is f0.
% 
% The three things I'd double-check if this is code you're maintaining or adapting:
% 
% The InstFirRate(numel(ts)+1:end) = [] line ties the length of the rate trace to the number of spikes, which isn't the usual way to align a rate signal to a time base — it's worth confirming that's intended rather than a leftover.
% 
% The switch lower('lsBinned') never varies, so the raw and robust fitting branches never execute, and the lsCoeff it computes is discarded in favour of CorrelateXYAndRegression. If you ever want to switch fitting methods, that hard-coded string is the thing to parameterise.
% 
% The comments about "EEG freq," "theta," and the beta/f0 oscillatory model are inherited from a theta-frequency version of this analysis; here everything operates on firing rate, so read those comments as historical context rather than a description of what's being computed.
% 
% If you tell me what nancorr and CorrelateXYAndRegression return (or paste them), I can verify the ret fields line up and flag any mismatches.