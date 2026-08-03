function [posx,posy,posts,mapAxis,visited]=posdata(PosMtx,threshold,shape,binWidth)

% eliminate coordinate zero from position
PosMtx(PosMtx(:,2)==0,:) = [];
PosMtx(PosMtx(:,3)==0,:) = [];
posx = PosMtx(1:end,2);
posy = PosMtx(1:end,3);
posts = PosMtx(1:end,1);

% % 
% % %coordonnees pour cellules directionnelles
% % posgx = PosMtx(:,2);
% % posgy = PosMtx(:,3);
% % % posrx = circshift(PosMtx(:,2),-1);
% % % posry = circshift(PosMtx(:,3),-1);
% % posrx = PosMtx(:,4);
% % posry = PosMtx(:,5);


% correct the x y coordinates 
% [posx,posy,posts] = remJumpTrack(posx,posy,posts,threshold);
% remove = find(posy>160);
% posx(remove) = [];
% posy(remove) = [];
% posts(remove) = [];

% [posx,posy,posts] = interpolatePos(posx,posy,posts);
%  then smooth and center
[posx,posy] = smooth_path(posx,posy);
[posx,posy] = center_path(posx,posy,shape);

[mapAxis] = mapaxis(posx,posy,binWidth);
visited = visitedBins(posx,posy,mapAxis); % calculate the visited zones of the arena 