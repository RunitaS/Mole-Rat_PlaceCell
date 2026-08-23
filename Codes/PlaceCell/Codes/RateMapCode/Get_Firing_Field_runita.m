function [placefield,placefield2] = Get_Firing_Field_runita(meanfiringmap,map,Num_of_fields_flag)
% function [placefield] = Get_Firing_Field_tiffany(map,Num_of_fields_flag)

% this function extracts the firing field based on a non smoothed rate map
% see fenton et all 1998 neurobiology  for further details
% place field is determined by 2 rules
% 1- spikes in bin > 0
% 2- share.s at least a side with other bins which contain spikes
% if num of field flag is set to 1 then it only finds the largest field
if nargin < 3
    Num_of_fields_flag =20;
end
grps1 = [];
% first increase size of map to remove edge effects
fieldmap = zeros(size(map)+2);
fieldmap(2:size(fieldmap,1)-1,2:size(fieldmap,2)-1) = map;
%map = [zeros(size(map,1),1),map,zeros(size(map,1),1)];
%map = [zeros(size(map,2),1)';map;zeros(size(map,2)',1)];
fieldmap2 = zeros(size(fieldmap));
group = 1;
for I=2:size(fieldmap,2)-1
    for J=2:size(fieldmap,1)-1
        if fieldmap(J,I) > meanfiringmap
%         if fieldmap(J,I) > 0
            if fieldmap(J+1,I) > 0 || fieldmap(J,I+1) > 0  || fieldmap(J-1,I) > 0  || fieldmap(J,I-1) > 0 
                  
                fieldmap2(J,I) = -1;
            end
        end
    end
end
map = map;
[J,I] = find(fieldmap2 == -1,1);
fieldmap2(J,I) = 1;
%for I=2:size(fieldmap,2)-1
%for J=2:size(fieldmap,1)-1
% group pixels that share at least a side togther 

while ~isempty(I)
    if (fieldmap2(J+1,I) == -1 && fieldmap2(J,I) ~= -1) || (fieldmap2(J,I+1) == -1 && fieldmap2(J,I) ~= -1)
    if fieldmap2(J+1,I) == -1 && fieldmap2(J,I) ~= -1;
        fieldmap2(J+1,I) = fieldmap2(J,I);
        J = J+1;
    elseif fieldmap2(J,I+1)== -1 && fieldmap2(J,I) ~= -1;
        fieldmap2(J,I+1) = fieldmap2(J,I);
        I = I+1;end
     else
        
        
        group = group +1;
        [J,I] = find(fieldmap2 == -1,1);
        if isempty(I)
            break;end
        fieldmap2(J,I) = group;
    end
    
    
    if fieldmap2(J-1,I) > 0 && fieldmap2(J,I) ~= fieldmap2(J-1,I);
        [ky,kx] = find(fieldmap2 == fieldmap2(J,I));
        for h=1:length(ky)
        fieldmap2(ky(h),kx(h)) = fieldmap2(J-1,I);end
    end
        
         %J = J-1;
     if fieldmap2(J,I-1) > 0 && fieldmap2(J,I) ~= fieldmap2(J,I-1);
        [ky,kx] = find(fieldmap2 == fieldmap2(J,I));
        for h=1:length(ky)
        fieldmap2(ky(h),kx(h)) = fieldmap2(J,I-1);end
     end
         %fieldmap2(J,I) = fieldmap2(J,I-1);
         %I = I-1;
    if fieldmap2(J+1,I) > 0 && fieldmap2(J,I) ~= fieldmap2(J+1,I);
        [ky,kx] = find(fieldmap2 == fieldmap2(J,I));
        for h=1:length(ky)
            fieldmap2(ky(h),kx(h)) = fieldmap2(J+1,I);end
    end
        
         %J = J-1;
     if fieldmap2(J,I+1) > 0 && fieldmap2(J,I) ~= fieldmap2(J,I+1);
        [ky,kx] = find(fieldmap2 == fieldmap2(J,I));
        for h=1:length(ky)
        fieldmap2(ky(h),kx(h)) = fieldmap2(J,I+1);end
     end
         %fieldmap2(J,I) = fieldmap2(J,I-1);
end
% if there is more than one field then find the largest
placefield = fieldmap2;
placefield2 = fieldmap2;

groups = unique(fieldmap2);
max_group_length = 0;
max_group_ind = 0;
group_length =[];
for I=2:length(groups)
    [ky,kx] = find(fieldmap2 == groups(I));
%     if length(ky) > max_group_length
%         max_group_length = length(ky);
%         max_group_ind = groups(I);
%     end
    group_length(groups(I)) = length(ky);
end
placefield = zeros(size(fieldmap2)-2);
[Y,IY] = sort(group_length,[2],'descend');
hi =0;
if ~isempty(group_length)
    if Num_of_fields_flag
        for IS =1:length(Y)
        if hi == 0;
            grp_ln = Y(IS); max_group_ind = IY(IS);
        else 
            break;
        end
        if grp_ln >= 9
            %for k=1:length(grps)
            hi  = 0;
             [ky,kx] = find(fieldmap2 == max_group_ind);
              for kky=1:length(ky)
                  %for kkx=1:length(kx)
                      if ky(kky)+2 <= size(placefield,1) && kx(kky)+2 <= size(placefield,2)
                          if ~any(fieldmap2(ky(kky):ky(kky)+2,kx(kky):kx(kky)+2) == 0)
                              
                              %grps1(hi) = grps(k);
                              hi = hi+1;
                              break
                              
                          end
                      end
                  %end
              end
            if hi >= 1
                for I=1:length(ky)
                    placefield(ky(I)-1,kx(I)-1) = 1;
                end
            else
                placefield = zeros(size(placefield));
            end
        end
     end
    else
        hi = 1;
        [ln,grps] = find(group_length > 9);
        for k=1:length(grps)
             [ky,kx] = find(fieldmap2 == grps(k));
              for kky=1:length(ky)
                  %for kkx=1:length(kx)
                      if ky(kky)+2 <= size(placefield,1) && kx(kky)+2 <= size(placefield,2)
                          if ~any(fieldmap2(ky(kky):ky(kky)+2,kx(kky):kx(kky)+2) == 0)
                              
                              grps1(hi) = grps(k);
                              hi = hi+1;
                              break
                              
                          end
                      end
                  %end
              end
        end
            
            
        if ~isempty(grps1)
            for I=1:length(grps1)
                [ky,kx] = find(fieldmap2 == grps1(I));
                for k=1:length(ky)
                    placefield(ky(k)-1,kx(k)-1) = I;
                end
            end
        end
    end
end
        

        
    

            
        