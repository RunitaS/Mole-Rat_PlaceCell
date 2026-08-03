function [ratemap] = ratemap_gaussian(h,spkx,spky,posx,posy,posts,binWidth,mapAxis)

% This function calculate a rate map from the matrix trajectory with
% spikes, by estimating the density of spikes in each position and applying
% a smoothing factor equal to 2*binWidth (defined in build_ratemap_gaussian
% program). It recall one external function: rate_estimator.

% from CBM, june 2011

invh = 1/h;
ratemap = zeros(length(mapAxis),length(mapAxis));
yy = 0;
for y = mapAxis
    yy = yy + 1;
    xx = 0;
    for x = mapAxis
        xx = xx + 1;
        ratemap(yy,xx) = rate_estimator(spkx,spky,x,y,invh,posx,posy,posts); % Calculate the rate for one position value
%         ratemap(yy,xx)=1;
    end
end

