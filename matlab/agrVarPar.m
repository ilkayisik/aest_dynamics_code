function [varPartition] = agrVarPar(data,spearFlag)

% agrVarPar.m
%
% function to compute variance partitioning of agreement data, such as on
% preference ratings. Based on H?nekopp 2006.
%
% NOTE: there MUST be repeat presentations of the same image to compute
% repeatable variance!
%
% INPUT:
%   data    [m,n,r] matrix of data where 
%                 m is number of items
%                 n is number of judges
%                 r is number of repetitions per judge
%
%   [spearFlag]   1 to use Spearman correlation, 0 for Pearson (default)
%                
%  OUTPUT:
%     varPartition    structure with the following fields:
%        .R           proportion of "repeatable" variance
%        .ST          proportion of total variance that is "shared" across judges 
%        .SR          proportion of repeatable variance that is "shared" across judges
%        .IT          proportion of total variance that is "individual" (not shared)
%        .IR         proportion of repeatable variance that is "individual" (not shared)
%        
% REVISION HISTORY:
%   2017-04-06  ev  added support for Spearman correlation
%   2015-10-17  ev  wrote it

[m, n, r] = size(data);

if nargin < 2
    spearFlag = 0;
    typeString = 'Pearson';
else
    if spearFlag
        typeString = 'Spearman';
    else
        typeString = 'Pearson';
    end
end

% repeatable variance:
% intra-participant correlation
for i = 1:n
    temp_corr = corr(squeeze(data(:,i,:)),'type', typeString);
    within_agr(i) =  mean(squareform(triu(temp_corr,1) + tril(temp_corr,-1)));
end
% square each; take the mean
% this is REPEATABLE var
varPartition.R = mean(within_agr.^2);

% compute inter-participant correlation across all possible pairs of
% individuals
subjAvgData = mean(data,3);
pairwise_corr_mat = corr(subjAvgData,'type',typeString);
pairwise_corr_list = squareform(triu(pairwise_corr_mat,1) + tril(pairwise_corr_mat,-1))';

% square each correlation; take the mean
% this is SHARED as a proportion of TOTAL
varPartition.ST = mean(pairwise_corr_list.^2);

% remaining is measurement error plus individual


% IT: INDIVIDUAL but REPEATABLE as proportion of TOTAL = R - ST
varPartition.IT = varPartition.R - varPartition.ST;
% SR = ST / R
varPartition.SR = varPartition.ST / varPartition.R;
% IR = IT / R
varPartition.IR = varPartition.IT / varPartition.R;

