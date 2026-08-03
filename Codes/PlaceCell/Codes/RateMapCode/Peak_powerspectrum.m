function [peak_freq1,peak_freq2,Z_power_spectrum,freq1,freq2] = Peak_powerspectrum(power_spectrum,Fs,freq1,freq2);

% calculate peak power spectrum at 2 freqs (ex theta and delta)
% freq1 and freq2 range of frequencies (ex: [4-10] for theta)

% column 2 : energy in each freq band (from 0 to Fs/2 and mirror)
% NB: LFP_FFT contains NFFT/2+1 points (see myFFT.m) that correspond to frequencies from zero to Fs/2
% find 5Hz

% normalize FFT (Z score - (x-mean)/sd)
Z_power_spectrum=(power_spectrum-mean(power_spectrum))/std(power_spectrum);

freq1_lim1 = freq1(1)*round(length(Z_power_spectrum)/(Fs/2));
freq1_lim2 = freq1(2)*round(length(Z_power_spectrum)/(Fs/2));
freq2_lim1 = freq2(1)*round(length(Z_power_spectrum)/(Fs/2));
freq2_lim2 = freq2(2)*round(length(Z_power_spectrum)/(Fs/2));

[peak_freq1,ind1]=max(Z_power_spectrum(freq1_lim1:freq1_lim2));
ind1=ind1+freq1_lim1;
freq1=ind1*Fs/2/round(length(Z_power_spectrum));
[peak_freq2,ind2]=max(Z_power_spectrum(freq2_lim1:freq2_lim2));
ind2=ind2+freq2_lim1;
freq2=ind2*Fs/2/round(length(Z_power_spectrum));