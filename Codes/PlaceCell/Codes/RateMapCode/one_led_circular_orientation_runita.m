function [deg_theta2] = one_led_circular_orientation_runita(PosMtx,shape)


samplerate = 30;
%coordinates for head direction
posgx2 = PosMtx(:,2);
posgy2 = PosMtx(:,3);

posrx2 = zeros(length(posgx2),1);
posry2 = zeros(length(posgx2),1);

posts = PosMtx(:,1);
posts = floor(posts/1000); %change for milisecond
posts = mod(posts,10000000); %change for milisecond
% then smooth
[posgx2,posgy2] = smooth_path(posgx2,posgy2);
[posgx2,posgy2] = center_path(posgx2,posgy2,shape);



thetacartesian2 = mod((180/pi)*(atan2(-posry2+posgy2, posrx2-posgx2)),360);
deg_theta2 = round(thetacartesian2);