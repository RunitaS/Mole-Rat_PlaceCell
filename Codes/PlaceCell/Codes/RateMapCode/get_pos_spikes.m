% this function calculate the coordinates xy of each spike. It is used to
% plot a spikemap (trajectory with spikes).

function [spkx,spky] = get_pos_spikes(ts,posx,posy,posts);


% *********************************************************
%              Build trajectory with spike map
% *********************************************************

% get the position of the spikes
N = length(ts);
spkx = zeros(N,1);
spky = zeros(N,1);
for i = 1:N
    [val,ind] = min(abs(posts-ts(i)));
    spkx(i) = posx(ind);
    spky(i) = posy(ind);
end

