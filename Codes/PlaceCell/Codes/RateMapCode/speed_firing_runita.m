function [ret,beta,f0]=speed_firing_runita(posx,posy,posts,ts,PosSampleRate)

speed = [];
for o = 1:length(posx)-1
    speed(o) = sqrt(((posx(o+1,1)-posx(o,1)).^2)+((posy(o+1,1)-posy(o,1)).^2));
end
% PosSampleRate = 25;

speed = speed*PosSampleRate;
speed = speed';
speed(end+1,1)=speed(end,1);

InstSpeed = speed;
Alltime = [posts(1):1000/PosSampleRate:posts(end)];
[FirRate,x]= histc(ts,Alltime);
FirRate = FirRate * PosSampleRate;

Speedsmoothingwindows = 0.08;
G = fspecial('gaussian',[Speedsmoothingwindows/PosSampleRate^-1,1],4);
InstFirRate = filtfilt(G,1,FirRate);
InstFirRate(numel(ts)+1:end) = [];

minspeed = 4; %(cm/s)
maxspeed = 100;
speedBinLengthCm = [2];
speedBins = [minspeed:speedBinLengthCm:maxspeed+speedBinLengthCm];
fit2bins = [1,numel(find(speedBins<maxspeed+speedBinLengthCm))];
ToCompare = find(InstSpeed>minspeed & InstSpeed<maxspeed);
ToCompare(find(ToCompare > numel(InstFirRate)))=[];
RawPearsonCorrelation = nancorr(InstSpeed(ToCompare),InstFirRate(ToCompare));

if isnan(RawPearsonCorrelation)
    ret.r = NaN;
    ret.p_value = NaN;
    ret.slope = NaN;
    ret.intercept = NaN;
%     ret = [];
    beta = [];
    f0 = [];
else
%%
speed2 = InstSpeed(ToCompare);
instFreq2 = InstFirRate(ToCompare);
nBins=length(speedBins)-1;
[nPerBin, binInd]=histc(speed2, speedBins); %Bin and get number of items per bin
nPerBin         =nPerBin(1:end-1); %Lose end bin out of range
dataPtsPerBin=nPerBin(1:end-1); %Last value of nPerBin is values matching last edge - do not need this

meanFreqPerBin=zeros(nBins,1);
stdFreqPerBin=zeros(nBins,1);
binCentres=zeros(nBins,1);
binSpeed = zeros(nBins,1);
semFreqPerBin = zeros(nBins,1);
for nnBin=1:nBins
    meanFreqPerBin(nnBin)=nanmean(instFreq2(binInd==nnBin)); %mean
    stdFreqPerBin(nnBin)=nanstd(instFreq2(binInd==nnBin)); %standard dev
    binCentres(nnBin)=nanmean([speedBins(nnBin); speedBins(nnBin+1)]);
    
    binSpeed(nnBin)=nanmean(speed2(binInd==nnBin)); %mean
    semFreqPerBin(nnBin) = stdFreqPerBin(nnBin)/sqrt(nPerBin(nnBin));
end
MaxSpeedBin = speedBins(end-1);
MaxNumberOfBins = numel(find(binCentres<MaxSpeedBin));
meanFreqPerBin(MaxNumberOfBins+1:end)=[];
stdFreqPerBin(MaxNumberOfBins+1:end)=[];
binCentres(MaxNumberOfBins+1:end)  =[];
semFreqPerBin(MaxNumberOfBins+1:end)  =[];

fit2bins(2) =MaxNumberOfBins;

%errorbar(binCentres,meanFreqPerBin,semFreqPerBin)
%%% FIT REGRESSION LINE TO VELOCITY VS EEG FREQ PLOT %%%%%%%%%%%%%%%%%%%%%
% Aim to deterimin slope of line and y-axis intersect (under oscillatory
% model these are called Beta and F0 respectivly).
% NB only use a subset of points to make this fit - very low speeds will be
% contamintated with non-movement related theta and high speeds are very
% under sampled.


%Use regress to do least squares best fit - seems that regress can also
%give the y-axis crossing (c) if the first set of y values it is gives is a
%column of ones, second value returned is Beta where y= Beta.x + c
%hence lsCoeff is [2 x 1] where lsCoeff(1) is c
switch lower('lsBinned')
    case 'lsbinned'
        lsCoeff = regress(meanFreqPerBin(fit2bins(1):fit2bins(2)), ...
            [ones(fit2bins(2)-fit2bins(1)+1,1) binCentres(fit2bins(1):fit2bins(2))]);
        
    case 'lsraw'
        %Decide which points are in range
        lowerSpeedLimit=speedBins(fit2bins(1));
        upperSpeedLimit=speedBins(fit2bins(2)+1);
        validSpeed=speed>=lowerSpeedLimit & speed<upperSpeedLimit;
        lsCoeff = regress(instFreq(validSpeed), [ones(sum(validSpeed),1), speed(validSpeed)]);
        
    case 'robustraw'
        %Decide which points are in range
        lowerSpeedLimit=speedBins(fit2bins(1));
        upperSpeedLimit=speedBins(fit2bins(2)+1);
        validSpeed=speed>=lowerSpeedLimit & speed<upperSpeedLimit;
        lsCoeff=robustfit(speed(validSpeed),instFreq(validSpeed));
end
PercentageDataPtsPerBin = dataPtsPerBin/sum(dataPtsPerBin)  ;
meanFreqPerBinToRegress = meanFreqPerBin;
meanFreqPerBinToRegress(find(PercentageDataPtsPerBin < (1/100) ) )    = NaN ;

in.x = binCentres;
in.y =meanFreqPerBinToRegress;
in.PLOT_ON = 1;
in.xlabel = 'cm/s';
in.ylabel = 'Rate (Hz)';
in.title = 'Speed correlation';

[ ret ] = CorrelateXYAndRegression(in);
beta = ret.slope;
f0   = ret.intercept;
% beta=lsCoeff(2);
% f0=lsCoeff(1);

%And for comparsion use the best fit line to estimate theta freq at
%15cm/s
freqAt15cm = (15*beta) + f0;

end

%Not using it at the moment but the matlab function robustfit will also do
%linear fitting but using somethign more sophisticated than least squares
%(ls is suseptable to outliers). Again returns a [2 x 1]
%NB doesn't seem to work too well
% rbCoeff = robustfit(trial(nEEG).meanFreq(lowerBin:upperBin), ...
%     trial(nEEG).binMidPt(lowerBin:upperBin));

end













    