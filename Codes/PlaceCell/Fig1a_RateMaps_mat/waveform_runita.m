function [waveform_el_1, waveform_el_2, waveform_el_3, waveform_el_4]= waveform_runita(waveforms)

% waveform = TTMtx2(TTMtx2(:,3)==cell(i),4:131);

waveform_el_1 = [];
waveform_el_2 = [];
waveform_el_3 = [];
waveform_el_4 = [];

for ii = 1:size(waveforms,1) 
    waveform_el_1(ii,:) = waveforms(ii,1:(128/4));
    waveform_el_2(ii,:) = waveforms(ii,(128/4)+1:(128/2));
    waveform_el_3(ii,:) = waveforms(ii,(128/2)+1:((128/2)+(128/4)));
    waveform_el_4(ii,:) = waveforms(ii,((128/2)+(128/4))+1:128);
%     waveform_el_1(ii,:) = waveforms(ii,1:4:125);
%     waveform_el_2(ii,:) = waveforms(ii,2:4:126);
%     waveform_el_3(ii,:) = waveforms(ii,3:4:127);
%     waveform_el_4(ii,:) = waveforms(ii,4:4:128);
end

average_waveform_el_1 = mean(waveform_el_1);
average_waveform_el_2 = mean(waveform_el_2);
average_waveform_el_3 = mean(waveform_el_3);
average_waveform_el_4 = mean(waveform_el_4);
average_all = [average_waveform_el_1; average_waveform_el_2; average_waveform_el_3; average_waveform_el_4];

y= peak2peak(average_all,2);
[row, col] = max(y);



selected_el = [];
for ii = 1:size(waveforms,1)
    if col == 1
    selected_el(ii,:) = waveforms(ii,1:(128/4));
    elseif col == 2
    selected_el(ii,:) = waveforms(ii,(128/4)+1:(128/2));
    elseif col == 3
    selected_el(ii,:) = waveforms(ii,(128/2)+1:((128/2)+(128/4)));
    elseif col == 4
    selected_el(ii,:) = waveforms(ii,((128/2)+(128/4))+1:128);
    end
end

average_selected_el = mean(selected_el);
std__selected_el = std(selected_el);


%%
std_plus_selected_el = average_selected_el+std__selected_el;
std_minus_selected_el = average_selected_el-std__selected_el;

derive_selected_el = [];
for iii = 1:size(selected_el,1)
    for iiii = 1:31
    derive_selected_el(iii,iiii) = (selected_el(iii,iiii+1)-selected_el(iii,iiii))/(1/32000);
    end
end
average_derive_selected_el = mean(derive_selected_el);
std__derive_selected_el = std(derive_selected_el);
std_plus_derive_selected_el = average_derive_selected_el(1:31)+std__derive_selected_el;
std_minus_derive_selected_el = average_derive_selected_el(1:31)-std__derive_selected_el;

%%
plot(average_selected_el,'linewidth',3,'color','k'); axis square;
hold on
plot(std_plus_selected_el,'--','linewidth',3,'color','k'); axis square;
plot(std_minus_selected_el,'--','linewidth',3,'color','k'); axis square;
hold off
% axis off
box off
% title('Mean waveform','fontsize',12);
xlim([0 32]);
set(gca, 'XTickLabel', [])
set(gca, 'XTick', [])
set(gca, 'YTickLabel', [])
set(gca, 'YTick', [])

% subplot(3,4,2)
% plot(average_derive_selected_el,'linewidth',3,'color','k'); axis square;
% hold on
% plot(std_plus_derive_selected_el,'--','linewidth',3,'color','k'); axis square;
% plot(std_minus_derive_selected_el,'--','linewidth',3,'color','k'); axis square;
% hold off
% % box off
% title('Mean derived waveform','fontsize',12);
% xlim([0 32]);
% set(gca, 'XTickLabel', [])
% set(gca, 'XTick', [])
% set(gca, 'YTickLabel', [])
% set(gca, 'YTick', [])
% 
% % subplot(3,4,3)
% % title('phase plot','fontsize',12);
% % line([0 0],[-(max(average_derive_selected_el)) max(average_derive_selected_el)],'linewidth',1,'color','k');
% % hold on
% % line([-(max(average_selected_el)) max(average_selected_el)],[0 0],'linewidth',1,'color','k');
% % plot(average_selected_el(1:31),average_derive_selected_el,'linewidth',3,'color','k'); axis square;
% % hold off
% % axis square
% average_selected_el_v2 = average_selected_el/(max(average_selected_el));
% average_derive_selected_el_v2 = average_derive_selected_el/(max(average_derive_selected_el));
% 
% subplot(3,4,3)
% title('Phase plot with centroid (red dot)','fontsize',12);
% polyin = polyshape(average_selected_el_v2(1:31),average_derive_selected_el_v2);
% % plot(polyin.Vertices(:,1),polyin.Vertices(:,2));
% [x,y] = centroid(polyin);
% line([0 0],[-1 1],'linewidth',1,'color','k');
% hold on
% line([-1 1],[0 0],'linewidth',1,'color','k');
% plot(polyin,'facecolor',[.7 .7 .7]);
% hold on
% scatter(x,y,20,'ro','filled')
% plot(average_selected_el_v2(1:31),average_derive_selected_el_v2,'--','linewidth',2,'color','k'); axis square;
% hold off
% axis square
% xlim([-1 1]);
% ylim([-1 1]),
% 
