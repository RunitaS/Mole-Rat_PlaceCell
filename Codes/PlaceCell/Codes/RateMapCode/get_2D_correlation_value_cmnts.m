function [corrcoeff,nonZero_corrcoeff,RrankO] = get_2D_correlation_value(A,B)
% GET_2D_CORRELATION_VALUE  Compute three correlation measures between two 2D matrices A and B.
%
%   Outputs:
%     corrcoeff          - standard Pearson correlation (via corr2), NaNs ignored
%     nonZero_corrcoeff  - Pearson correlation computed only over positions where
%                          at least one of A or B is non-zero
%     RrankO             - Spearman (rank) correlation, NaNs ignored
%
%   If only A is supplied, B defaults to A (autocorrelation, which is always 1).
%   The header comment claims it returns -99 for all-zero input, but the code
%   below actually returns -2 (see the else-branch).

% ---------------------------------------------------------------------------
% 1) Handle the single-input case
% ---------------------------------------------------------------------------
if nargin < 2            % nargin = how many inputs the caller actually passed
    B = A;               % only one matrix given -> compare A against itself
end

% ---------------------------------------------------------------------------
% 2) Sanity-check that the two matrices are the same size
% ---------------------------------------------------------------------------
if size(A) ~= size(B)    % element-wise compare of the size vectors [rows cols]
    % NOTE (latent bug): `if` on a vector is TRUE only when EVERY element is
    % non-zero. size(A)~=size(B) is a 1x2 vector, so this error only fires when
    % BOTH the row count AND the column count differ. A mismatch in just one
    % dimension slips through unchecked.
    error('\n Matrices and A and B should be of equal sizes')
end

% ---------------------------------------------------------------------------
% 3) Initialize the write index for the "non-zero" arrays built later
% ---------------------------------------------------------------------------
k = 1;                   % running index into A_nonzero / B_nonzero

% ---------------------------------------------------------------------------
% 4) Standard Pearson correlation, with NaNs removed
% ---------------------------------------------------------------------------
% ~isnan(A) & ~isnan(B) is a logical mask that is TRUE only at positions where
% BOTH matrices hold a valid (non-NaN) number. Indexing A(mask) / B(mask)
% linearizes the kept elements into column vectors. corr2 then returns their
% Pearson correlation coefficient (covariance / product of std deviations).
rorg = corr2(A(~isnan(A) & ~isnan(B)),B(~isnan(A) & ~isnan(B)));

% ---------------------------------------------------------------------------
% 5) Spearman rank correlation, with NaNs removed
% ---------------------------------------------------------------------------
% Same NaN masking as above, but corr(...,'type','Spearman') first converts the
% values to their ranks and correlates those. This measures the strength of a
% monotonic (not necessarily linear) relationship between A and B.
Rrank = corr(A(~isnan(A) & ~isnan(B)),B(~isnan(A) & ~isnan(B)),'type','Spearman');

% ---------------------------------------------------------------------------
% 6) Replace any remaining NaNs with zeros before the loop
% ---------------------------------------------------------------------------
A(isnan(A)) = 0;         % every NaN entry in A is overwritten with 0
B(isnan(B)) = 0;         % every NaN entry in B is overwritten with 0

% ---------------------------------------------------------------------------
% 7) Collect every (A,B) pair where at least one of the two is non-zero
% ---------------------------------------------------------------------------
for I=1:size(A,1)                     % loop over rows       (size(A,1) = #rows)
for J=1:size(A,2)                     % loop over columns    (size(A,2) = #cols)
if A(I,J) ~= 0 || B(I,J) ~= 0         % keep the pair if EITHER value is non-zero
            A_nonzero(k) = A(I,J);    % store A's value (array grows on each hit;
                                      %   not preallocated -> slower for big data)
            B_nonzero(k) = B(I,J);    % store B's matching value at the same index
            k = k+1;                  % advance the write index for the next hit
end
end
end

% ---------------------------------------------------------------------------
% 8) Pearson correlation computed manually over the collected non-zero pairs
% ---------------------------------------------------------------------------
if exist('A_nonzero','var') && exist('A_nonzero','var')   % (typo: checks A_nonzero
                                                          %  twice; harmless here)
% Manual Pearson formula:
%   r = sum((a - mean(a)).*(b - mean(b)))
%       -------------------------------------------------
%       sqrt( sum((a-mean(a)).^2) .* sum((b-mean(b)).^2) )
% i.e. covariance numerator divided by the product of the (unnormalized) spreads.
r = sum((A_nonzero - mean(A_nonzero)).*(B_nonzero - mean(B_nonzero)))/(sqrt(sum((A_nonzero - mean(A_nonzero)).^2).*sum((B_nonzero - mean(B_nonzero)).^2)));
else
    r = -2;              % A_nonzero never got created -> every element was zero:
    Rrank = -2;          % flag both correlations with the sentinel value -2
end

% ---------------------------------------------------------------------------
% 9) Assign outputs according to how many the caller asked for
% ---------------------------------------------------------------------------
if nargout >= 1          % nargout = number of outputs requested by the caller
    corrcoeff = rorg;    % 1st output: standard Pearson correlation
end
if nargout >= 2
    nonZero_corrcoeff = r;   % 2nd output: non-zero-only Pearson correlation
end
if nargout >= 2          % NOTE: this should almost certainly be `>= 3`; as written
    RrankO = Rrank;      % the 3rd output is assigned whenever >=2 are requested
end


%%%
% It takes two 2D matrices A and B of equal size and returns three different correlation measures between them:
% 
% corrcoeff — the ordinary Pearson correlation (linear relationship), computed with MATLAB's corr2, ignoring NaNs.
% nonZero_corrcoeff — a Pearson correlation computed only over the positions where at least one of A or B is non-zero. This is the "ignore the zeros" idea: positions that are zero in both matrices are dropped, so a shared background of zeros doesn't inflate the correlation.
% RrankO — the Spearman rank correlation (monotonic relationship), again ignoring NaNs.
% How it flows
% If you pass only one matrix, it sets B = A, giving an autocorrelation (which is trivially 1).
% It builds a mask ~isnan(A) & ~isnan(B) — true only where both matrices are valid numbers — and uses it to compute the Pearson (corr2) and Spearman (corr(...,'Spearman')) values on clean data.
% It then replaces leftover NaNs with 0, and loops over every cell. Whenever A(I,J) or B(I,J) is non-zero, it copies that pair into A_nonzero/B_nonzero.
% On those collected pairs it applies the Pearson formula by hand: covariance numerator over the product of the spreads.
% Finally it hands back only as many outputs as the caller asked for.
% 
% The difference between measure 1 and measure 2 is the mask: measure 1 keeps every non-NaN cell (including cells that are 0 in both), while measure 2 discards the both-zero cells before correlating.
% 
% A few quirks worth knowing about (all noted inline in the file)
% 
% These aren't things you asked me to fix, but they affect the actual computation, so they're worth flagging:
% 
% The size check is weak. if size(A) ~= size(B) evaluates a 1×2 vector, and an if on a vector is only true when every element is non-zero. So the error fires only when both dimensions differ — a mismatch in just rows or just columns passes silently.
% The exist check has a typo: exist('A_nonzero','var') && exist('A_nonzero','var') tests the same variable twice (the second was probably meant to be B_nonzero). Harmless here, since both are created together.
% The all-zero sentinel is -2, not -99. The header comment says it returns -99 when everything is zero, but the else branch actually sets r = -2 and Rrank = -2.
% The third output is guarded by nargout >= 2 instead of >= 3. As written, RrankO gets assigned whenever two or more outputs are requested, which is probably not intended.
% A_nonzero/B_nonzero aren't preallocated, so they grow one element at a time inside the loop — fine for small matrices, slow for large ones.
% 
