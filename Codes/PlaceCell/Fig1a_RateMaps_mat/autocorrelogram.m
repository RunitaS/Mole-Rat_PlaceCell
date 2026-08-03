function [corr,y] = autocorrelogram(TimeWindow,Tbin,ts)

%% compute time autocorrelogram
% TimeWindow = 1000;   % length of the autocorr
% Tbin = 5;           % bins of the autocorr
% times1 = OpenField1.tetrode(Tetrode_cell1).ts(find(OpenField1.tetrode(Tetrode_cell1).cut==Cell_1))  ; %
% times1 = times1*1000;    
times1 = ts; %
% times1 = times1*1000;    
% subplot(2,2,2);
[corr,y] = calcXCH_TimeWindow (times1,times1, TimeWindow,Tbin);
bar(y,corr,1);%hold on
% plot(y,corr)
% set(get(gca,'Children'),'FaceColor',[0.6 0.6 0.6]);
% set(get(gca,'Children'),'EdgeColor',[0 0 0]);
set(get(gca,'Children'),'FaceColor','k');
set(get(gca,'Children'),'EdgeColor','k');
set(gca,'TickDir','out','Box','off');

%set(get(gca,'XLabel'),'Fontsize',25,'String','Interval (ms)');
%set(get(gca,'YLabel'),'Fontsize',25,'String','N spikes');
set(gca,'ylim',[-1,max(corr)*1.2]);
% xlim([0 (TimeWindow + (TimeWindow*0.1)) ])
xlim([-(TimeWindow+(TimeWindow*0.1)) (TimeWindow + (TimeWindow*0.1)) ])
ylims = get(gca,'ylim');
ylim([0 ylims(2)]);
line([0 0],[0 ylims(2)],'Color',[0 0 0]);
% ylim([0 1]);
% line([0 0],[0 1],'Color',[0 0 0]);
axis square
    h=get(gcf, 'currentaxes');
set(h, 'fontsize', 20, 'linewidth', 10);
box off
get(gca,'ticklength');
set(gca,'TickLength',[0.02,0.01]);
set(gca,'TickDir','out');
% xlabel('Lags (msec)');
% ylabel('Autocorrelation');
% set(gca,'Ytick',[0:150:300]);
