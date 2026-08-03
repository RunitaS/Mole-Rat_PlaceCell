function [z,r] = coherence(rmap)
% Calculates coherence from visited pixels only.
% BRIAN

   N = double(rmap>=0);                       % Visited pixel template.
   rmap(rmap<0) = 0;                          % Cleaned rate map.
   h = ones(1,3);                             % Convolution kernel.

   N2 = conv2(h, h, N, 'same') - N;           % Number of pixels to average.
   rmap2 = conv2(h, h, rmap, 'same') - rmap;  % Total rate around each pixel.
   
   N2 = N2(N>0);                              % Discard unvisited pixels.
   rmap = rmap(N>0);                          %            |
   rmap2 = rmap2(N>0);                        %            -
   N2(N2==0) = inf;                           % Isolated pixels will be correlated with 0.
   rmap2 = rmap2 ./ N2;                       % Average rate around each pixel.
   rho = cov(rmap,rmap2);                     % Covariance. Normalises by N-1.
   r = rho(1,2) / sqrt(rho(1,1)*rho(2,2));    % Correlation coefficient.
   
   %N = sum(sum(N));                           % Total number of pixels.
   %z = sqrt(N-3) * 1/2 * log((1+r)/(1-r));    % Z transform. Normalised version (for probability calculation).
   z = 1/2 * log((1+r)/(1-r));                % Z transform. Un-normalised version (Muller & Kubie, 1989).
