function [time_map] = timemap(mapAxis,posx,posy,posts,freq_sampl,bins);

%This function allows to build a spike map from the coordinate position of
% each spike
%bins=length(mapAxis);
time_map = zeros(bins,bins);
N = length(posts);
for i = 1:N
    ind_x = max(find(mapAxis<=posx(i)));
    ind_y = max(find(mapAxis<=posy(i)));
    time_map(ind_y,ind_x) = time_map(ind_y,ind_x)+1;
end
% convert the number of visits in time spent (seconds)
% position sampling is 40hz (i.e. 0.025 sec)
% time_map = time_map*(1/freq_sampl);
time_map = time_map*freq_sampl/1000;
