% function  polarplot_v2(OpenfIeld,posgx,posgy,posrx,posry,posts,ts,threshold,shape)
function [deg_theta,posdirts]= polarplot_runita(posx,posy,PosMtx,ts,correction,binSizeDir,spkx,spky)
%polar coordinates
samplerate = 30;

%coordinates for head direction
posgx = PosMtx(:,2);
posgy = PosMtx(:,3);
posrx = [];
posry = []; 
posrx = circshift(posgx,-1);
posry = circshift(posgy,-1);

posts = PosMtx(:,1);
posts = floor(posts/1000); %change for milisecond
posts = mod(posts,10000000); %change for milisecond
% then smooth
[posgx,posgy] = smooth_path(posgx,posgy);
% [posgx,posgy] = center_path(posgx,posgy,shape);
posts = PosMtx(:,1);
posts = floor(posts/1000); %change for milisecond
posts = mod(posts,10000000); %change for milisecond
[posrx,posry] = smooth_path(posrx,posry);
% [posrx,posry] = center_path(posrx,posry,shape);


thetacartesian = mod((180/pi)*(atan2(-posry+posgy, posrx-posgx)),360);
deg_theta = round(thetacartesian);

posdirts = zeros(length(posts),1);
posdirts(1,1) = sum(ts > 0 & ts <= posts(1,1));
for l = 2:size(posdirts,1)
    posdirts(l,1) = sum(ts > posts(l-1,1) & ts <= posts(l,1));
end

degbin=0:binSizeDir:360;
dirmap = zeros(size(degbin,2)-1,2);
for b = 1:size(degbin,2)-1
    dirmap(b,1) = sum(deg_theta > degbin(b) & deg_theta <= degbin(b+1));
    dirmap(b,2) = sum(posdirts(deg_theta > degbin(b) & deg_theta <= degbin(b+1),1));
end

dirrate=dirmap(:,2)./(dirmap(:,1)/samplerate);
dirrate(isnan(dirrate))=0;
dirrate = smooth(dirrate,3);

%correction of animal head relative to LEDs
dirrate = circshift(dirrate,-correction/binSizeDir);

Angles = [0:binSizeDir:360-binSizeDir]';
nbins = numel(Angles);
Rate = [dirrate];
x = cosd(Angles).*Rate;
y = sind(Angles).*Rate;

MaxBinRate = find(Rate == max(Rate)) ; 
cc = colormapc(5,nbins);
% polar plot in blue
ColorForMaxPeak = [0 0 1]   ; % This is Blue but you can change to whatever colour you specify from colormapc function
% ColorForMaxPeak = [1 0 0] ; %red color
% ColorForMaxPeak = [0 1 0] ; %green color
% ColorForMaxPeak = [0 1 0];
% ColorForMaxPeak = [1 0 0];

MaxBinColorMap = find(ismember(cc,ColorForMaxPeak,'rows'))   ;MaxBinColorMap=MaxBinColorMap(1);
ToShift = MaxBinRate -  MaxBinColorMap ;
cc = circshift(cc,ToShift);

%%
subplot(1,2,1)
line([0 0],[-(max(Rate)) max(Rate)],'linewidth',2,'color','k');
hold on
line([-(max(Rate)) max(Rate)],[0 0],'linewidth',2,'color','k');
X = smooth(x,5);
Y = smooth(y,5);
 
% X = (x);
% Y = (y);
hold on ;
% v = [x y];
v = [X Y];
f = (1:1:length(dirrate));
col = cc;
patch('Faces',f,'Vertices',v,'FaceVertexCData',col,'EdgeColor','interp','FaceColor','non','LineWidth',8,'LineSmoothing','off');

hold off;
axis equal tight;
axis off;
%%
subplot(1,2,2);

Angles = [0:binSizeDir:360-binSizeDir]';
plot(posx,posy,'Color',[.6 .6 .6],'Linewidth',3);axis square; axis off ; set(gca,'DataAspectRatio',[1 1 1]);

hold on; 
SpikePos = [spkx spky];
scatter(SpikePos(:,1),SpikePos(:,2));  


dir = get_pos_spikes(ts,deg_theta,deg_theta,posts);

DirbinInd = zeros(numel(dir),1);
    nPerBin = [];
dirBins = Angles;
    for iBin = 1 : numel ( dirBins )
     binInd = [];
       if iBin ~= numel(dirBins)
           Bin = [dirBins(iBin)  dirBins(iBin+1)];
           if Bin(1) > Bin(2)
               binInd = [ binInd ; find(dir > Bin(1) ); find( dir < Bin(2)) ] ;
           else
               binInd = [ binInd ; find(dir > Bin(1) & dir < Bin(2)   ) ] ;
           end
           nPerBin(iBin,1) = numel(binInd);
           DirbinInd(binInd) = iBin;
       else
           Bin = [dirBins(iBin)  dirBins(1)];
           if Bin(1) > Bin(2)
               binInd = [ binInd ; find(dir > Bin(1) ); find( dir < Bin(2)) ] ;
           else
               binInd = [ binInd ; find(dir > Bin(1) & dir < Bin(2)   ) ] ;
           end
           nPerBin(iBin,1) = numel(binInd);
           DirbinInd(binInd) = iBin ;
       end
       hold on
       SpikesToPlot = find(DirbinInd == iBin);a = 150;
%        scatter(posx(SpikePos(SpikesToPlot),1),posy(SpikePos(SpikesToPlot),2) ,'MarkerFaceColor', [cc(iBin,:)],'MarkerEdgeColor', [cc(iBin,:)] );
       scatter(SpikePos(SpikesToPlot,1),SpikePos(SpikesToPlot,2) ,a,'MarkerFaceColor', [cc(iBin,:)],'MarkerEdgeColor', [cc(iBin,:)] );
       hold off
       SpikesToPlot=[];
    end
end


