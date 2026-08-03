function [instfreq,hx] = hilbert_instfreq(x,Fs);

% this function calculate the instantaneous frequency of filtered LFP using
% hilbert transform. hilbert transforms give the analytic signal (i.e. the enveloppe)
% and instfreq corresponds to the difference in phase between each time
% point
hx = hilbert(x);
instfreq = Fs/(2*pi)*diff(unwrap(angle(hx)));