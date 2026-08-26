% placefield identifies the placefields in the map. It returns the
% number of placefields and the location of the peak within each
% placefield.
function [nFields,fieldPos,centreFieldSize] = placefield(map,pTreshold,pBins,Axis)
Axis = binWidth * linspace(-((size(acorrmap,1)-1)/2),((size(map,1)-1)/2),size(map,1));
% Minimum number of bins in a placefield. Fields with less than pBins bins
% are not defined as a placefield. (Bin size are 1.5cm x 1.5cm)
%pBins = 100;
%pTreshold = 0.1;
centreFieldSize = NaN;
centreX = 1000;
centreY = 1000;
binWidth = Axis(2) - Axis(1);
nFields = 0;
fieldPos = [];
peak = max(max(map));
% Allocate memory to the arrays
[M,N] = size(map);
% Array that contain the bins of the map this algorithm has visited
visited = zeros(M,N);
% Find the bins that have rate below the treshold
index = find(map < pTreshold);
% Set bins with rate below treshold to visited.
visited(index) = 1;
visited(isnan(map)) = 1;
% Go as long as there are unvisited parts of the map left
while ~prod(prod(visited))
% Array that will contain the bin positions to the current placefield
binsX = [];
binsY = [];
% Find the unvisited bins in the map
[I,J] = find(visited==0);
% The first unvisited bin are set as the starting point
legalI = I(1);
legalJ = J(1);
% Go as long as there still are bins left in the current placefield
while 
1
% Add current bin to the bin position arrays
binsX = [binsX; legalI(1)];
binsY = [binsY; legalJ(1)];
% Set the current bin to visited
visited(legalI(1),legalJ(1)) = 1;
% Find which of the 4 neigbour bins that are part of the placefield
[legalI,legalJ] = getLegals(visited,legalI,legalJ);
% Remove current bin from the array containing bins to be added
legalI(1) = [];
legalJ(1) = [];
% Check if we are finished with this placefield
if 
length(legalI)==0
break;
end
end
if 
length(binsX)>=pBins % Minimum size of a placefield
nFields = nFields + 1;
% Find centre of mass (com)
comX = 0;
comY = 0;
% Total rate
R = 0;
for 
ii = 1:length(binsX)
R = R + map(binsX(ii),binsY(ii));
comX = comX + map(binsX(ii),binsY(ii))*Axis(binsX(ii));
comY = comY + map(binsX(ii),binsY(ii))*Axis(binsY(ii));
end
% Check if this is the centre field
if 
sqrt((comX/R)^2+(comY/R)^2) < sqrt(centreX^2+centreY^2)
centreX = comX/R;
centreY = comY/R;
centreFieldSize = length(binsX) * binWidth^2;
end
fieldPos = [fieldPos; [comX/R, comY/R]];
end
end
