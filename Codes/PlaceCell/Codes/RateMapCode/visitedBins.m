function visited = visitedBins(posx,posy,mapAxis); 
% This function calculates what area of the map that has been visited by 
% the rat.
% from CBM, june 2011


% Number of bins in each direction of the map
N = length(mapAxis);
% Number of position samples
M = length(posx);
visited = zeros(N);

for ii = 1:N
    for jj = 1:N
        px = mapAxis(ii);
        py = mapAxis(jj);
        distance = sqrt( (px-posx).^2 + (py-posy).^2 );
        
        if min(distance) <= 3
            visited(jj,ii) = 1;
        end
    end
end