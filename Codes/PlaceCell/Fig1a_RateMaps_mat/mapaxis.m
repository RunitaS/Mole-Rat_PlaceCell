function [mapAxis] = mapaxis(posx,posy,binWidth)
% this function sets the axis for ratemap
% Fra november 2015 from CBM

% get the lenght of the recorded map (the real length is 150cm).
obsSLength = max(max(posx)-min(posx),max(posy)-min(posy));
bins = ceil(obsSLength/binWidth); % gets the N bins in x y axis of the map
sLength = binWidth * bins; % calculate the corrected length
mapAxis = (-sLength/2+binWidth/2):binWidth:(sLength/2-binWidth/2); % set map axis

