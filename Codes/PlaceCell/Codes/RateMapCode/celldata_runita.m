function  celldata_runita(inputfile,tetrodes,pp,cells,iiii,waveforms,ts,posx,posy,posts,visited,smooth_factor,binWidth,mapAxis,TimeWindow,Tbin,diam_arena,PosMtx,correction,binSizeDir)   
figure;
%% mean waveform, mean derived waveform and phase plot   
    subplot(2,4,1);
    [waveform_el_1, waveform_el_2, waveform_el_3, waveform_el_4]=waveform_runita(waveforms);
    set(gca,'TickDir','out','Box','off');
    h=get(gcf, 'currentaxes');
    set(h, 'fontsize', 12, 'linewidth', 2);
    set(gca,'TickLength',[0.02,0.01]);

waveforms2 = [waveform_el_1, waveform_el_2, waveform_el_3, waveform_el_4];
RecordingDuration = posts(end,1)/1000;
spikeDuration = 1;
NumOfPoints = 32;
SpikesFigure = 0;

[peak,troughs,PeakTroughRatio,avg_freq,width,widthAHP,peak2trough,endslope,AmplitudeRSD]=waveform_properties(waveforms2,ts,spikeDuration,NumOfPoints,RecordingDuration,SpikesFigure);


%% show interval interspike for refractory period
    subplot(2,4,2);
    IsI = diff(ts); % get the diff between spikes time
    bins =0:1:10;	% bining of distribution for 10 msec
    histogram(IsI,bins); % plot distribution of IS
    set(gca,'xtick',(0:2:50));
    set(get(gca,'Children'),'FaceColor',[0.6 0.6 0.6]);
    set(get(gca,'Children'),'EdgeColor',[0 0 0]);
    set(gca,'TickDir','out','Box','off');
    xlabel('Msec');
    ylabel('Spikes');
    set(gca,'TickDir','out','Box','off');
    h=get(gcf,'currentaxes');
    set(h,'fontsize', 12,'linewidth',2);
    set(gca,'TickLength',[0.02,0.01]);
    title('Interspike Intervals','fontsize',12);
    axis square

%% plot time map
    subplot(2,4,7);
    freq_sampl = 30;
    bins = length(mapAxis);
    [time_map] = timemap(mapAxis,posy,posx,posts,freq_sampl,bins);
    maxtimemap = max(time_map(:));
    HH = fspecial('disk',smooth_factor);
    time_map = imfilter(time_map,HH,'replicate');
    time_map(visited==0) = NaN;
    mytimefig=pcolor(time_map);
    colormap jet
    set(mytimefig,'Edgecolor','none')
    set(gca,'DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1]);      
    axis off        
    title(strcat('Time map, max=',num2str(sprintf('%.2f',maxtimemap)),'sec'),'fontsize',12);

%% plot trajectory with spikes   
    subplot(2,4,5);  
    [spkx,spky] = get_pos_spikes(ts,posx,posy,posts); % get the position of the spikes
    plot(posx,posy,'color',[.2 .2 .2],'linewidth',2);
    hold on
    scatter(spkx,spky,200,'.r');
    title('Path spikes');
    set(gca,'DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1]);
    axis off
    box off
    
%% plot rate map   
    h = smooth_factor*binWidth;
    [ratemap] = ratemap_gaussian(h,spkx,spky,posx,posy,posts,binWidth,mapAxis); % build matrix ratemap 
    % remove invisited bins from the rate map
    ratemap(visited==0) = NaN;
    peak = max(ratemap(:));
    peak = peak*1000;
    peak_2=sprintf('%.2f', peak);
    subplot(2,4,6);
    myfig=pcolor(ratemap);
    colormap jet
    set(myfig,'Edgecolor','none')
    title(strcat('Peak rate = ',(peak_2),'Hz'));
    set(gca,'DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1]);
    axis off

%%      
    if length(ts)>50
%% calculate and plot cell firing/speed relationship
    % % % % % subplot(3,4,9);
    % % % % % [ret,beta,f0]=speed_firing_runita(posx,posy,posts,ts,25);
    
%% calculate and plot time firing autocorrelagram
    subplot(2,4,3);
    [autocorr,y] = autocorrelogram(TimeWindow,Tbin,ts);
    set(gca,'TickDir','out','Box','off');
    h=get(gcf, 'currentaxes');
    set(h, 'fontsize', 12, 'linewidth', 2);
    set(gca,'TickLength',[0.02,0.01]);
    title('Autocorrelogram','fontsize',12);

%% calculate and plot intrinsic firing theta modulation
    subplot(2,4,4);
    [mean_freq,freq_peak_distr,peak_theta,peak_delta] = intrinsicfiring(TimeWindow,Tbin,autocorr,y);
    title(strcat('Peak theta=',num2str(sprintf('%.2f', freq_peak_distr),'Hz')),'fontsize',12);
    [peak_frequency_theta] = theta_delta(TimeWindow,Tbin,ts);
    set(gca,'TickDir','out','Box','off');
    h=get(gcf, 'currentaxes');
    set(h, 'fontsize', 12, 'linewidth', 2);
    set(gca,'TickLength',[0.02,0.01]);

    else
     peak_theta_frequency=[];Grid_score=[];ret=[];beta=[];f0=[];autocorr=[];y=[];mean_freq=[];freq_peak_distr=[];peak_theta=[];peak_delta=[];
    end
%% save figure
name = inputfile(1:end-4);
nametosave = strcat('D:\Résultats expériences\Projet Runita\Runita_OfflineSorter_20Oct23\',name,'_tet',num2str(tetrodes(pp)),'cell',num2str(cells(iiii)),'all');

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
set(gcf,'Position',[0 0 1 1]);
set(gcf,'PaperOrientation','landscape');
% print(nametosave,'-dpdf','-fillpage','-r0')
print(nametosave,'-dpng','-r0')
close()

    
    %%
time_session = (posts(end)-posts(1))/1000;
number_of_spikes = length(ts);
mean_session = number_of_spikes/time_session;
mean_ratemap = nanmean(ratemap(:))*1000;

%% spatial coherence
    h = 1;
    [mapraw] = ratemap_gaussian(h,spkx,spky,posx,posy,posts,binWidth,mapAxis); % build matrix ratemap 
    mapraw(visited==0) = NaN;
    rate_Nan=isnan(mapraw);
    mapraw(rate_Nan==1)=0;
    [z,r] = coherence(mapraw);
    spatial_coherence=r;

    %% skaggs or information content
rates= mapraw;
times = timemap(mapAxis,posy,posx,posts,freq_sampl,bins);

if(size( rates ) > 1)
    % turn arrays into column vectors
    rates = reshape( rates, prod(size(rates)), 1);
    times = reshape( times, prod(size(times)), 1);
end

duration = sum(times);
mean_rate = sum(rates.*times)./duration;

p_x = times./duration;
p_r = rates./mean_rate;                   
dum = p_x.*rates;        
ind = find( dum > 0 );   
bits_per_sec = sum(dum(ind).*log2(p_r(ind)));   % sum( p_pos .* rates .* log2(p_rates) )
bits_per_spike = bits_per_sec/mean_rate;
skags = bits_per_spike;
%% find placefield

threshold_placefield = max(ratemap(:))/2.5;
% threshold_placefield = mean_session/1000;
[placefield] = Get_Firing_Field_runita(threshold_placefield,ratemap);

placefield(visited==0) = NaN;
placefield_number = max(max(placefield));

figure;
myplacefield=pcolor(placefield);
% colormap jet
set(myplacefield,'Edgecolor','none');
set(gca,'DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1]);
axis off;
%% save figure place field
name = inputfile(1:end-4);
nametosave = strcat('D:\Résultats expériences\Projet Runita\Runita_OfflineSorter_20Oct23\',name,'_tet',num2str(tetrodes(pp)),'cell',num2str(cells(iiii)),'place field');

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
set(gcf,'Position',[0 0 1 1]);
set(gcf,'PaperOrientation','landscape');
% print(nametosave,'-dpdf','-fillpage','-r0')
print(nametosave,'-dpng','-r0')
close()
%% intrasession correlation
time_max_session = posts(end);
half_time_session = time_max_session-(time_session*1000/2);
ts_1st_half = ts(ts<half_time_session);
posts_1st_half = posts(posts<half_time_session);
posx_1st_half = posx(posts<half_time_session);
posy_1st_half = posy(posts<half_time_session);
[spkx_1st_half,spky_1st_half] = get_pos_spikes(ts_1st_half,posx_1st_half,posy_1st_half,posts_1st_half);

ts_last_half = ts(ts>half_time_session);
posts_last_half = posts(posts>half_time_session);
posx_last_half = posx(posts>half_time_session);
posy_last_half = posy(posts>half_time_session);
[spkx_last_half,spky_last_half] = get_pos_spikes(ts_last_half,posx_last_half,posy_last_half,posts_last_half);

h = smooth_factor*binWidth;
[ratemap_1st_half] = ratemap_gaussian(h,spkx_1st_half,spky_1st_half,posx_1st_half,posy_1st_half,posts_1st_half,binWidth,mapAxis);
visited_1st_half = visitedBins(posx_1st_half,posy_1st_half,mapAxis); ratemap_1st_half(visited_1st_half==0) = NaN;
[ratemap_last_half] = ratemap_gaussian(h,spkx_last_half,spky_last_half,posx_last_half,posy_last_half,posts_last_half,binWidth,mapAxis);
visited_last_half = visitedBins(posx_last_half,posy_last_half,mapAxis);ratemap_last_half(visited_last_half==0) = NaN;

[corrcoeff,nonZero_corrcoeff,Spearman] = get_2D_correlation_value(ratemap_1st_half,ratemap_last_half);
figure;
subplot(2,2,1);
    plot(posx_1st_half,posy_1st_half,'color',[.6 .6 .6],'linewidth',2);
    hold on
    scatter(spkx_1st_half,spky_1st_half,500,'.r');
    title('Path spikes 1st half');
    set(gca,'DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1]);
    axis off;
subplot(2,2,2);
    
    plot(posx_last_half,posy_last_half,'color',[.6 .6 .6],'linewidth',2);
    hold on
    scatter(spkx_last_half,spky_last_half,500,'.r');
    title('Path spikes last half');
    set(gca,'DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1]);
    axis off;
    
subplot(2,2,3);
    myfig=pcolor(ratemap_1st_half);
    set(myfig,'Edgecolor','none'); colormap jet;
    axis xy
    title('Ratemap 1st half');
    set(gca,'DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1]);
    axis off;
subplot(2,2,4);
    myfig=pcolor(ratemap_last_half);
    set(myfig,'Edgecolor','none')
    axis xy
    title('Ratemap last half');; colormap jet;
    set(gca,'DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1]);
    axis off;
%% save figure intrasession correlation
name = inputfile(1:end-4);
nametosave = strcat('D:\Résultats expériences\Projet Runita\Runita_OfflineSorter_20Oct23\',name,'_tet',num2str(tetrodes(pp)),'cell',num2str(cells(iiii)),'intrasession corr');

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
set(gcf,'Position',[0 0 1 1]);
set(gcf,'PaperOrientation','landscape');
% print(nametosave,'-dpdf','-fillpage','-r0')
print(nametosave,'-dpng','-r0')
close()
%% calculate and plot polar plot
figure;
[deg_theta,posdirts]= polarplot_runita(posx,posy,PosMtx,ts,correction,binSizeDir,spkx,spky);
%% save figure polar plot
name = inputfile(1:end-4);
nametosave = strcat('D:\Résultats expériences\Projet Runita\Runita_OfflineSorter_20Oct23\',name,'_tet',num2str(tetrodes(pp)),'cell',num2str(cells(iiii)),'polar plot');

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
set(gcf,'Position',[0 0 1 1]);
set(gcf,'PaperOrientation','landscape');
% print(nametosave,'-dpdf','-fillpage','-r0')
print(nametosave,'-dpng','-r0')
close()
%% Direction by time   
figure;
subplot(4,1,[2 3]);
shape = 'circle';
[deg_theta2] = one_led_circular_orientation_runita(PosMtx,shape);
% x_axis = 1:length(deg_theta);
x_axis = posts;
jumps = find(abs(diff(deg_theta2)) > 354) + 1;
% plot(x_axis(1:jumps(1)-1),deg_theta(1:jumps(1)-1),'color',[.6 .6 .6], 'linewidth', 2);
plot(deg_theta2(1:jumps(1)-1),x_axis(1:jumps(1)-1),'color',[.6 .6 .6], 'linewidth', 2);


for t = 2:length(jumps)
    hold on
%     plot(x_axis(jumps(t-1):jumps(t)-1),deg_theta(jumps(t-1):jumps(t)-1),'color',[.6 .6 .6], 'linewidth', 2);%[110 101 101]/255)
    plot(deg_theta2(jumps(t-1):jumps(t)-1),x_axis(jumps(t-1):jumps(t)-1),'color',[.6 .6 .6], 'linewidth', 2);%[110 101 101]/255)

end
% plot(x_axis(jumps(t):end),deg_theta(jumps(t):end),'color',[.6 .6 .6], 'linewidth', 2);
plot(deg_theta2(jumps(t):end),x_axis(jumps(t):end),'color',[.6 .6 .6], 'linewidth', 2);
hold on
% scatter(x_axis(posdirts>0),deg_theta(posdirts>0),'filled','MarkerEdgeColor','k','MarkerFaceColor','r')
scatter(deg_theta2(posdirts>0),x_axis(posdirts>0),'filled','MarkerEdgeColor','k','MarkerFaceColor','r')
xlim([0 360]);
set(gca,'xtick',0:90:360);
box off;
set(gca,'TickDir','out','Box','off');
h=get(gcf, 'currentaxes');
set(h, 'fontsize', 12, 'linewidth', 2);
set(gca,'TickLength',[0.02,0.01]);
ylabel('Time (sec)');
xlabel('Head direction (°)');

[distribution_time,~] = histcounts(deg_theta2,'BinLimits',[0, 360],'binwidth',binSizeDir);
distribution_time = distribution_time*freq_sampl/1000;
distribution = deg_theta2(posdirts>0);
[distribution_spike,~] = histcounts(distribution,'BinLimits',[0, 360],'binwidth',binSizeDir);

rate = distribution_spike./distribution_time;

X1 = repmat(rate,1,3);
X2 = [];
for t= length(rate)+1:length(rate)*2
    X2(t-length(rate)) = mean([X1(t-1) X1(t) X1(t+1)]);
end

linear_ratemap = [X2 ; NaN(1,length(X2))];
subplot(4,1,4);
fig2 = pcolor(linear_ratemap);
colormap jet
set(fig2,'Edgecolor','none')
axis xy
axis square
set(gca,'DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',[1 1 1]);
axis off


subplot(4,1,1);
bar(distribution_spike)
xlim([0 60]);
box off;

%% save figure directionXtime
name = inputfile(1:end-4);
nametosave = strcat('D:\Résultats expériences\Projet Runita\Runita_OfflineSorter_20Oct23\',name,'_tet',num2str(tetrodes(pp)),'cell',num2str(cells(iiii)),'directionXtime');

set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
set(gcf,'Position',[0 0 1 1]);
set(gcf,'PaperOrientation','landscape');
% print(nametosave,'-dpdf','-fillpage','-r0')
print(nametosave,'-dpng','-r0')
close()

end