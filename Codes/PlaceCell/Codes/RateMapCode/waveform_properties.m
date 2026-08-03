function [peak,troughs,PeakTroughRatio,avg_freq,width,widthAHP,peak2trough,endslope,AmplitudeRSD]=waveform_properties(waveforms,ts,spikeDuration,NumOfPoints,RecordingDuration,SpikesFigure)

% This function extracts general features of the spikes waveform

% INPUTS
% waveforms= matrix in the format [spikes X sampled points] where each line is a single action potential and the columns the voltage values sampled for each tetrode. 
%            The four tetrodes must be concatenated so that you have (time1_tetrode1...timeN_tetrode1;time1_tetrode2...timeN_tetrode2;...;time1_tetrode4...timeN_tetrode4);
% spikeDuration= length of the time sampled around each spike;
% NumOfPoints= number of sampled points for each spike;
% clusterID= UNIQUE ID for each isolated neuron. Mmake sure that it is really unique to save figures;
% RecordingDuration= duration of the session (used to calculate the average frequency);
% savepath=path where you want to save figures;
% SpikeFigure= 1 if you want the figures;

% OUTPUTS
% peak=average amplitude of the spikes;
% troughs= average depth of the AHP;
% PeakTroughRatio= ration between the amplitude of the spikes and its AHP;
% avg_freq= average frequency of the neuron;
% width= half-height width of the spikes;
% widthAHP= half-depth width of the AHP. NaN if it's impossible to identify the trough;
% peak2trough= peak to trough time;
% endslope= slope of the waveform 10bins (should be about 500us) after the trough (used to identify PV+ neurons in Niell and Stryker 2008; confirmed by optotagging in Moore and Wehr 2013). NaN if it's impossible to identify the trough;
% AmplitudeRSD= variation coefficient (or relative standard deviation) of the amplitude of the spikes;

% % 


peak=[];
troughs=[];
PeakTroughRatio=[];
avg_freq=[];
width=[];
peak2trough=[];
widthAHP=[];
endslope=[];
AmplitudeRSD=[];
spikesNumber=length(waveforms(:,1));

% zztop = figure('visible', 'off');
for channel=1:4
    if SpikesFigure==1
        subplot(2,2,channel)
        for spike=1:length(waveforms(:,1))
            a=plot(waveforms(spike,1+(NumOfPoints*(channel-1)):NumOfPoints+(NumOfPoints*(channel-1))));
            set(a,'Color',[0.3 0.3 0.3]); axis square;
            hold on
        end
        ch=waveforms(1:spikesNumber,1+(NumOfPoints*(channel-1)):NumOfPoints+(NumOfPoints*(channel-1)));
        avg_spikes=mean(ch,1);
        CoefficientVariation_ch(channel)=std(max(ch,[],2))/mean(max(ch,[],2));
        only_avg(channel,:)=avg_spikes;
        peak_ch(channel)=max(avg_spikes);
        b=plot(avg_spikes);
        set(b,'Color',[1 0 0]);
%         print(strcat(savepath,filename),'-djpeg','-r800');
%         close(gcf);
    else
        ch=waveforms(1:spikesNumber,1+(NumOfPoints*(channel-1)):NumOfPoints+(NumOfPoints*(channel-1)));
        avg_spikes=mean(ch,1);
        CoefficientVariation_ch(channel)=std(max(ch,[],2))/mean(max(ch,[],2));
        only_avg(channel,:)=avg_spikes;
        peak_ch(channel)=max(avg_spikes);
    end
end
% print(strcat(savepath,filename),'-djpeg','-r200');
% close(zztop);

    [~,principal_ch]=max(peak_ch);
    shape=only_avg(principal_ch,:);
    AmplitudeRSD=CoefficientVariation_ch(principal_ch);
    avg_freq=length(ts)/RecordingDuration;
    [peak,peak_idx]=max(shape);
    Aftershape=shape(peak_idx:end);
    [troughs,hyper_idx] =min(Aftershape); 
    PeakTroughRatio=peak./troughs;
    peak2trough=hyper_idx*(spikeDuration/NumOfPoints);
    [pk,~,w,~]=findpeaks(shape,'MinPeakProminence',15);
    [tr,loctr,whyp,~]=findpeaks((-(shape(peak_idx:end))),'MinPeakProminence',5);
    [~,idx]=max(pk);
    [~,idxtr]=max(tr);
    width=(w(idx)*(spikeDuration/NumOfPoints));

%     sometimes it's impossible to identify a trough because it's flat
    if isempty(idxtr)
        widthAHP=NaN;
        endslope=NaN;
    else
        widthAHP=(whyp(idxtr)*(spikeDuration/NumOfPoints));
        endslope=(Aftershape(loctr(1)+10)-Aftershape(loctr(1)))/((spikeDuration/NumOfPoints)/9);
%         endslope=NaN; % l89-90 comment and replace by NaN by PYves because we don't have all the waveform
    end
%     widthAHP=(whyp(idxtr)*(spikeDuration/NumOfPoints));
%     endslope=(Aftershape(loctr(1)+10)-Aftershape(loctr(1)))/((spikeDuration/NumOfPoints)/9);
    
end
