function [mean_freq,freq_peak_distr,peak_theta,peak_delta] = intrinsicfiring(TimeWindow,Tbin,corr,y)

Fs = 1000/Tbin;            % Sampling frequency                    
[LFP_FFT] = myFFT(corr,Fs);
power_spectrum=(LFP_FFT(:,2));
% take peak power theta and delta (normalized power)
theta_freq = [4 12];
delta_freq = [2 4];

[peak_theta,peak_delta,Z_power_spectrum] = Peak_powerspectrum(power_spectrum,Fs,theta_freq,delta_freq);
% figure
[c index4] = min(abs(LFP_FFT(:,1)-4));
[c index12] = min(abs(LFP_FFT(:,1)-12));
plot(LFP_FFT(:,1),Z_power_spectrum,'color','k','linewidth',3);
set(gca,'TickDir','out','Box','off');
h=get(gcf, 'currentaxes');
set(h, 'fontsize', 12, 'linewidth', 2);
set(gca,'TickLength',[0.02,0.01]);
xlim([LFP_FFT(index4,1) LFP_FFT(index12,1)]);
xlabel('Frequency (Hz)');
ylabel('Amplitude');
threshold = 5*mean(Z_power_spectrum);
line([LFP_FFT(index4,1) LFP_FFT(index12,1)],[threshold threshold],'color','r','linestyle','--','linewidth',3);

%%%%%%%%%%%%%%%%% FREQUENCY for entire session
% calculate instantaneous freq - Hilbert transform
% lfp=cat(2,y,corr);
lfp=cat(1,y,corr)';
lfp_theta = FilterLFP(lfp,Fs/2,'passband','theta');
[inst_freq,hx] = hilbert_instfreq(lfp_theta(:,2),Fs);
mean_freq=mean(inst_freq); % average freq
[counts centers]=hist(inst_freq,min(inst_freq):1:max(inst_freq));
[val ind]=max(counts);
freq_peak_distr=centers(ind);


