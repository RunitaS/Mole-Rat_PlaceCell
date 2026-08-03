% This function allow to find the centre of the recording box. it is used
% to correct the path
function [NE,NW,SW,SE] = centreBox(posx,posy)

% Find border values for path and box
maxX = max(posx);
minX = min(posx);
maxY = max(posy);
minY = min(posy);

% Set the corners of the reference box
NE = [maxX, maxY];
NW = [minX, maxY];
SW = [minX, minY];
SE = [maxX, minY];

