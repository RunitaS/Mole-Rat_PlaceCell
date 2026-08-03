%% Find and list the neuralynx files (ntt)
FilesList=dir('*.ntt');
Files = struct2table(FilesList);
t = 1;
tt = 0;
%nttFile = 'C:\Runita\NMR\Data\FA8477\Open_Field\Day8_ExperimentDay5_19Aug22\2022-08-19-13-53-55\TT4\TT4_SS_01_SS_02.ntt'; %your reocrding file

coordinates = readmatrix('C:\Runita\NMR\Data\FA8477\SAB\1D\Tracking\Tracking_forRateMaps\18Aug22_Session4_CleanTracking.xlsx');% coordinates of position

pathfigure = 'C:\Runita\NMR\Temp'; % folder to save figure (finish the path with\)


%% Choose tetrode + cell (if need)
tetrode = []; %[] if you want all cells or write the number if you want a specific cell
cell = [];   % write number of cell only if you specified a number of tetrode, else, write []
NbOfTetrodsFiles = length(FilesList(:,1)); % amount of tetrod files (ntt)

%define paramaters:
diam_arena = 80; % diameter of the circular track
binWidth = 5;     % bining for ratemaps
smooth_factor = 5;  % factor of smoothing for ratemap
TimeWindow = 1000;  % length of time autocorrelogram (in msec)
Tbin = 10;          % bin of autocorrelogram (in msec)        
threshold = 100;    % for path correction
binSizeDir = 6;    % bining for polar plot
correction = 0;     % angles shift of LEDs position on the animal head
shape = 'circle';

%% Extract and assemble spike features from all .ntt files

for f = 1%:NbOfTetrodsFiles
    inputfile = 'TT1_0003_SS_28.ntt';

    [TTMtx] = ExtractNlxSpike(inputfile);
end

%% Extract and assemble positions (PosMtx) and angles coordinates (Angle)
% [PosMtx, Angle] = ExtractNlxPos(nvtFile);
% PosMtx(:,1) = PosMtx(:,1)/1000;%conversion de microSec vers milliSec

PosMtx(:,1) = coordinates(:,1); %position timestamp
PosMtx(:,2) = coordinates(:,2); % x coordinates
PosMtx(:,3) = coordinates(:,3); % y coordinates

if ~isempty(tetrode)
    
    TTMtx2=TTMtx(TTMtx(:,2)== tetrode,:); %extract all spike info for a tetrode
    [posx,posy,posts,mapAxis,visited]=posdata(PosMtx,threshold,shape,binWidth); % position processing (smoothing, center, preparing axis matrix for ratemap, visited bins...)
    posts = floor(posts/1000); %change for milisecond
    posts = mod(posts,10000000); %change for milisecond
    
    cells = unique(TTMtx2(:,3));

    ts = TTMtx2(TTMtx2(:,3)==cell,1); %extract all spike info for a cluster/cell
    ts = floor(ts/1000); %change for milisecond
    ts = mod(ts,10000000); %change for milisecond
    waveforms = TTMtx2(TTMtx2(:,3)==cell,4:131); %extract all info for waveform   
    tetrodes = unique(TTMtx(:,2));
    pp = find(tetrodes == tetrode);
    cells = unique(TTMtx2(:,3));
    iiii = find(cells == cell);
    celldata_runita(inputfile,tetrodes,pp,cells,iiii,waveforms,ts,posx,posy,posts,visited,smooth_factor,binWidth,mapAxis,TimeWindow,Tbin,diam_arena,PosMtx,correction,binSizeDir); %analysis function

else
    tetrodes = unique(TTMtx(:,2));
    for pp = 1:length(tetrodes)
        TTMtx2=TTMtx(TTMtx(:,2)== tetrodes(pp),:);
        TTMtx2(TTMtx2(:,3)==0,:) = [];
        [posx,posy,posts,mapAxis,visited]=posdata(PosMtx,threshold,shape,binWidth);
        posts = floor(posts/1000);
        posts = mod(posts,10000000);
        cells = unique(TTMtx2(:,3));

       if ~isempty(cells)

           for iiii = 1:length(cells)
               ts = TTMtx2(TTMtx2(:,3)==cells(iiii),1);
               ts = floor(ts/1000);
               ts = mod(ts,10000000);
               
               if length(ts)>1
                   waveforms = TTMtx2(TTMtx2(:,3)==cells(iiii),4:131);
                   celldata_runita(inputfile,tetrodes,pp,cells,iiii,waveforms,ts,posx,posy,posts,visited,smooth_factor,binWidth,mapAxis,TimeWindow,Tbin,diam_arena,PosMtx,correction,binSizeDir);    
               end
           end
       end
    end
end









