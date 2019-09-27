root = '/Users/ilkay.isik/aesthetic_dynamics/';
cd(root)
addpath('/Users/ilkay.isik/aesthetic_dynamics/aest_dynamics_code/matlab/');
rate_path = 'data/data_rate.mat';
view_path = 'data/data_view.mat';

% m = items, n = raters, r = repetition
m = 30;
n = 25;
r = 2;

rate = zeros(m, n, r);
view = zeros(m, n, r);

load(rate_path)
rate(:, :, 1) = oData_ses1';
rate(:, :, 2) = oData_ses2';
rate(16, 22, 2) = 0.9999; % this subject 22 has all lscp ratings 1, that's why it returns nan
clear oData_ses1 oData_ses2
load(view_path)
view(:, :, 1) = oData_ses1';
view(:, :, 2) = oData_ses2';
      
%     varPartition    structure with the following fields:
%        .R           proportion of "repeatable" variance
%        .ST          proportion of total variance that is "shared" across judges 
%        .SR          proportion of repeatable variance that is "shared" across judges
%        .IT          proportion of total variance that is "individual" (not shared)
%        .IR          proportion of repeatable variance that is "individual" (not shared)  
corr_type = 'Pearson';
varPar{1} = agrVarPar(rate(1:15, :, :), corr_type); % rate dance
varPar{2} = agrVarPar(rate(16:end, :, :),corr_type); % rate lscp
varPar{3} = agrVarPar(view(1:15, :, :), corr_type); % view dance
varPar{4} = agrVarPar(view(16:end, :, :), corr_type); % view lscp
qNames = {'Rate: dance','Rate: landscape', 'View: dance','View: landscape'};
nQ=4;

%Variance Partitioning
qColOffset = 0.5;
fCol = ([.3 .3 .3; .6 .6 .6; .3 .3 .3; .1 .1 .1]);
qColMask = [1 0 0; 0 1 0; 1 0 0; 0 1 0];

figure;
subplot(2,nQ,1);

for q = 1:nQ
    subplot(2,nQ,q);
    ph{q} = pie([varPar{q}.ST varPar{q}.IT 1-varPar{q}.R ], {'ST' 'IT' 'NR'});
    ph{q} = pie([varPar{q}.ST varPar{q}.IT 1-varPar{q}.R ], {num2str(varPar...
            {q}.ST), num2str(varPar{q}.IT), num2str(1-varPar{q}.R)});
    title(qNames(q));
    set(ph{q}(1),'FaceColor', 1 - fCol(1,:) .* (1-qColMask(q,:)));
    set(ph{q}(3),'FaceColor', 1 - fCol(2,:) .* (1-qColMask(q,:)));
    set(ph{q}(5),'Facecolor', 1 - fCol(3,:) .* (1-qColMask(q,:)));
     
    subplot(2,nQ,q+nQ);
    ph2{q}=pie([varPar{q}.SR varPar{q}.IR],{'SR' 'IR'});
    ph2{q}=pie([varPar{q}.SR varPar{q}.IR],{[num2str(round(varPar{q}.SR * ...
           100)), ' %'], [num2str(round(varPar{q}.IR * 100)), ' %']});
    set(gcf,'Color',[1 1 1]);
    
    set(ph2{q}(1),'FaceColor',1 - fCol(1,:) .* (1-qColMask(q,:)));
    set(ph2{q}(3),'FaceColor',1 - fCol(2,:) .* (1-qColMask(q,:)));
    
end


% Figure 6 (C): Only for 'SR' & 'IR'
% Color settings: 
qColOffset = 0.5;
fCol = ([.3 .3 .3; .6 .6 .6; .3 .3 .3; .6 .6 .6]);
qColMask = [1 0 0; 0 1 1; 1 0 0; 0 1 1];

figure;
set(gcf,'Color',[1 1 1]);
for q = 1:nQ
    subplot(1,nQ,q );
    
   
    
    ph2{q}=pie([varPar{q}.SR varPar{q}.IR],{'SR' 'IR'});
    % use percentage 
    ph2{q}=pie([varPar{q}.SR varPar{q}.IR],{[num2str(round(varPar{q}.SR * ...
           100)), ' %'], [num2str(round(varPar{q}.IR * 100)), ' %']});
   
   
    set(ph2{q}(1),'FaceColor', 1 - fCol(1, :) .* (1-qColMask(q,:)));
    set(ph2{q}(3),'FaceColor', 1 - fCol(2, :) .* (1-qColMask(q,:)));
    title(qNames(q));
end

saveas(gcf,'/Users/ilkay.isik/aesthetic_dynamics/output/figures/Fig_06_C_varpar.pdf')
