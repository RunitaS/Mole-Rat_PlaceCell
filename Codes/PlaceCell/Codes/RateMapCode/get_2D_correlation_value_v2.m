function [corrcoeff,nonZero_corrcoeff,RrankO] = get_2D_correlation_value(A,B)
% it calculation Correlation matrix for 2D in two ways 
% 1- normal correlation coeffiecent using matlab corr2
% 2- correlation while ignoring the zeros
% 3- rank correlations;
% returns -99 if all the eleements in the matrix are zero

if nargin < 2
    B = A; % if there is only a single inputcalculate the autocorrelation I guess
    
end
% check that A and B equal sizes 
if size(A) ~= size(B)
    error('\n Matrices and A and B should be of equal sizes')
end
% get the sums first A and B
k = 1;
rorg = corr2(A(~isnan(A) & ~isnan(B)),B(~isnan(A) & ~isnan(B)));
Rrank = corr(A(~isnan(A) & ~isnan(B)),B(~isnan(A) & ~isnan(B)),'type','Spearman');
A(isnan(A)) = 0;
B(isnan(B)) = 0;
%num = 0;
for I=1:size(A,1)
    for J=1:size(A,2)
        if A(I,J) ~= 0 || B(I,J) ~= 0
            A_nonzero(k) = A(I,J);
            B_nonzero(k) = B(I,J);
            %num = num +((A(I,J)-mean2(A))*(B(I,J)-mean2(B))
            k = k+1;
        end
    end
end
if exist('A_nonzero','var') && exist('A_nonzero','var')
r = sum((A_nonzero - mean(A_nonzero)).*(B_nonzero - mean(B_nonzero)))/(sqrt(sum((A_nonzero - mean(A_nonzero)).^2).*sum((B_nonzero - mean(B_nonzero)).^2)));

else 
    r = -2;
    Rrank = -2;
end
%rz = 0.5*(log(1+r)./(1-r))
%rorg = corr2(A,B);
if nargout >= 1
    corrcoeff = rorg;
end
if nargout >= 2
    nonZero_corrcoeff = r;
end
if nargout >= 2
    RrankO = Rrank;
end