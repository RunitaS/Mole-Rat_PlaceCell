function [fieldsLabels,FieldSizeReal,RateMean,RatePeak,FieldsCentroid,FieldSizeEst,FieldSizeMajAx,RedZoneCentroid,Ecc]=placefields_localmax(ratemap,binWidth,psizemin,psizemax,t)
% The function plots the position of the place fields starting from a rate map
 
% using the local maxima to establish the firing rate threshold. It is useful
% to analyze place fields in grid maps with an heteregeneous firing rate.
% It returns several parameters for each place field.
% % % % % % % % % % % % % % % % %INPUTS % % % % % % % % % % % % % % % % % % %
 
% ratemap=the ratemap possibly already smoothed.
% binwdth=size of the bins of the rate map in cm
% psizemin=minimum size of a place field in cm^2
% psizemax=maximum size of place fields in cm^2
% t= rate threshold for fields detection. Fraction of the local maxima. The HIGHER the t the LOWER the threshold
% % % % % % % % % % % % % % % % %OUTPUTS % % % % % % % % % % % % % % % % % % %
 
% fieldLabels= map of the firing fields. It can be used for a pcolor
% FieldSizeReal= real size of the fields calculated on the basis of the number of pixels
% RateMean= mean firing rate in the fields
% RatePeak= maximal firing rate in the fields
% FieldsCentroid= position (x,y) of the centroid of the fields calculated on the real field shapes
% FieldSizeEst=size of the fields calculated as the area of the ellipse circumscribing the fields. It include a correction
 
% to reduce the understimation of the size for the place fields cut by the border.
% FieldSizeMajAx= Area of the zone calculated on the basis of the major
% axis(thought to be compared with other methods);
% RedZoneCentroid= position (x,y) of the centroid calculated on the more active portion of the fields. It should
% reduce the effect of a possible distortion of the field on its position.
% Ecc=Eccentricity of the field. 0<Ecc<1; If Ecc=1 the field is a perfect circle.
 
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
 
pbinmax=sqrt(psizemax)/binWidth;
binArea=binWidth^2;
ratemap(isnan(ratemap))=0;
% find the local maxima
localmax = imregionalmax(ratemap); 
indmax=zeros(nnz(localmax),2);
indxmax=1;
for i=1:length(localmax(:,1))
row=localmax(i,:);
for 
j=1:length(row)
if 
row(j)==1
indmax(indxmax,:)=[i,j];
indxmax=indxmax+1;
end
end
end
% find all the bins around the local maximum with a frequency higher than the threshold
 
for k=1:length(indmax(:,1))
localthreshold=ratemap(indmax(k,1),indmax(k,2));
if 
(localthreshold)*1000<=1
continue
end
for 
i= indmax(k,1)-(ceil(pbinmax/2)):indmax(k,1)+(ceil(pbinmax/2))
if 
i<=0 || i> length(localmax)
continue
end
row=ratemap(i,:);
for 
j=indmax(k,2)-(ceil(pbinmax/2)):indmax(k,2)+(ceil(pbinmax/2))
if 
j<=0 || j>length(row)
continue
end
if 
ratemap(i,j)>=localthreshold/t
localmax(i,j)=1;
end
end
end
end
% Calculate all the parameters
fieldsLabels = bwlabel(localmax,8);
labelList = setdiff(unique(fieldsLabels),0);
FieldsCentroid=zeros(length(labelList),2);
FieldSizeReal=zeros(length(labelList),1);
RateMean=zeros(length(labelList),1);
RatePeak=zeros(length(labelList),1);
FieldSizeEst=zeros(length(labelList),1);
Centr_Per=zeros(length(labelList),1);
RedZoneCentroid=zeros(length(labelList),2);
AxisLength=zeros(length(labelList),1);
FieldSizeMajAx=zeros(length(labelList),1);
Ecc=zeros(length(labelList),1);
for 
i=1:length(labelList)
fieldTemp = fieldsLabels==labelList(i);
if 
sum(fieldTemp(:)*binArea) < psizemin % If field too small, continue
continue
end
FieldSizeReal(i,1)=sum(fieldTemp(:)*binArea);
centr=regionprops(fieldTemp,'Centroid');
FieldsCentroid(i,:)=horzcat(centr.Centroid(1),centr.Centroid(2));
RateMean(i)=mean(ratemap(find(fieldTemp)))*1000;
RatePeak(i)=max(ratemap(find(fieldTemp)))*1000;
hotspot=(RatePeak(i)/1000)/1.33;
hotmap=zeros(length(ratemap));
pix=regionprops(fieldTemp,'PixelIdxList');
for 
k=1:length(pix.PixelIdxList(:,1))
if 
ratemap(pix.PixelIdxList(k))>= hotspot
hotmap(pix.PixelIdxList(k))=1;
end
end
hotmap=bwlabel(hotmap,8);
RedCentroidTemp=regionprops(hotmap,'Centroid');
RedZoneCentroid(i,:)=RedCentroidTemp.Centroid;
minax=regionprops(fieldTemp,'MinorAxisLength');
majax=regionprops(fieldTemp,'MajorAxisLength');
AxisLength(i,1)=majax.MajorAxisLength;
EccTemp=regionprops(fieldTemp,'Eccentricity');
FieldSizeMajAx(i,1)=((majax.MajorAxisLength/2)^2)*pi*binArea;
if 
EccTemp.Eccentricity==0
Ecc(i,1)=0.0001;
else
Ecc(i,1)=(EccTemp.Eccentricity); 
end
if 
(majax.MajorAxisLength/minax.MinorAxisLength)>= 2
FieldSizeEst(i,1)= ((majax.MajorAxisLength/2)^2)*pi*binArea;
Centr_Per= 'Peripheral';
else
FieldSizeEst(i,1)= ((majax.MajorAxisLength/2)*(minax.MinorAxisLength/2)*pi)*binArea;
Centr_Per='Central';
end
end
FieldsCentroid=[nonzeros(FieldsCentroid(:,1)) nonzeros(FieldsCentroid(:,2))];
FieldsCentroid=(ceil(FieldsCentroid) -(length(ratemap)/2))*binWidth;
FieldSizeReal=nonzeros(FieldSizeReal);
RateMean=nonzeros(RateMean);
RatePeak=nonzeros(RatePeak);
FieldSizeEst=nonzeros(FieldSizeEst);
RedZoneCentroid=[nonzeros(RedZoneCentroid(:,1)) nonzeros(RedZoneCentroid(:,2))];
RedZoneCentroid=(ceil(RedZoneCentroid) -(length(ratemap)/2))*binWidth;
FieldSizeMajAx=nonzeros(FieldSizeMajAx);
Ecc=nonzeros(Ecc);
end
