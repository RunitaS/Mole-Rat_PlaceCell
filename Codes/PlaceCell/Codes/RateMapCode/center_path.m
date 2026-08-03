function [posx,posy] = center_path(posx,posy,shape);

% this function center the path in box or maze
% Fra, november 2015

if strcmp(shape,'maze')
    [NE,NW,SW,SE] = centreMaze;
else
    [NE,NW,SW,SE] = centreBox(posx,posy);
end
centre = findCentre(NW,SW,NE,SE);
% set all coordinated relative to the centre 
posx = posx-centre(1);
posy = posy-centre(2);