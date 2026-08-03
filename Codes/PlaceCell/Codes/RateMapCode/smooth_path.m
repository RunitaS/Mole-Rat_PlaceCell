function [posx,posy] = smooth_path(posx,posy)

% this function apply a moving average filter to the path

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
span = 14;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 for cc = span/2+1:length(posx)-span/2
     posx(cc) = mean(posx(cc-span/2:cc+span/2));
     posy(cc) = mean(posy(cc-span/2:cc+span/2));
 end