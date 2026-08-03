function r = rate_estimator(spkx,spky,x,y,invh,posx,posy,posts)

% This function calculate the rate for one position value by using a edge-
% corrected kernel density estimator. It uses an external function:
% gaussian_kernel. 

% from CBM, june 2011

conv_sum = sum(gaussian_kernel(((spkx-x)*invh),((spky-y)*invh)));
edge_corrector =  trapz(posts,gaussian_kernel(((posx-x)*invh),((posy-y)*invh)));
%edge_corrector(edge_corrector<0.15) = NaN;
r = (conv_sum / (edge_corrector + 0.0001)) + 0.0001; % regularised firing rate for "wellbehavedness"
                                                       % i.e. no division
                                                       % by zero or log of zero