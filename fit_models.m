function [results, bms_results] = fit_models(data,M)
    
    % Fit belief state TD models.
    %
    % USAGE: [results, bms_results] = fit_models(data)
    %
    % INPUTS:
    %   data - [S x 1] data structure, where S is the number of subjects,
    %          with the following fields:
    %           .N - total number of trials
    %           .r - [N x 1] rewards
    %           .trial - [N x 1] trial number
    %           .previous - [N x 1] previous block identity: small (0) and large (1) blocks
    %           .DA_US - [N x 1] DA signal at the time of outcome
    %           .DA_US_sd - [N x 1] DA signal at the time of outcome sem
    %           .lick - [N x 1] cue-evoked anticipatory licking
    %           .lick_sd - [N x 1] cue-evoked anticipatory licking sem
    %           .DA_CS - [N x 1] cue-evoked DA signal
    %           .DA_CS_sd - [N x 1] cue-evoked DA signal sem
    %           .v0 - [1 x 2] reward expectations for pre-trained small and large blocks
    %
    % OUTPUTS:
    %   results - see mfit_optimize for more details
    %
    % Sam Gershman, Aug 2016
    
    rng(1); % set random seed so results are reproducible
    
    % load data if none specified
    if nargin < 1
        load data_CSUSmean1s
    end

    try
        load results
    end
    
    % fit models
    models = {'noBS0' 'noBS2' 'BS0' 'BS1' 'BS2' 'BS2_multi'};  
    if nargin < 2; M = 1:length(models); end
    for m = M
        disp(['Fitting model ',models{m}]);
        param = getparam(models{m});
        f = @(x,data) likfun(x,data,models{m});
        results(m) = mfit_optimize(f,param,data);
    end
    
    bms_results = mfit_bms(results);
    
end

function param = getparam(model)
    
    param_alpha.name = 'alpha';
    param_alpha.logpdf = @(x) 0;  % uniform prior
    param_alpha.lb = 0.001;  % lower bound
    param_alpha.ub = 0.3;  % upper bound
    
    param_sd.name = 'sd';
    param_sd.logpdf = @(x) 0;
    param_sd.lb = 0.001;
    param_sd.ub = 0.5;
    
    param_b.name = 'b';
    param_b.logpdf = @(x) 0;
    param_b.lb = -0.5;
    param_b.ub = 0.5;
    
    param_p.name = 'p';
    param_p.logpdf = @(x) 0;
    param_p.lb = 0.001;
    param_p.ub = 0.999;
    
    param_v.name = 'v';
    param_v.logpdf = @(x) 0;
    param_v.lb = 0;
    param_v.ub = 1;
    
    switch model
        case 'noBS0'
            param(1) = param_alpha;
        
        case 'noBS2'
            param(1) = param_alpha;
            param(2) = param_v; param(2).name = 'v0';
            param(3) = param_v; param(2).name = 'v1';
            
        case {'BS0'}
            param(1) = param_alpha;
            param(2) = param_sd;
            
        case 'BS1'
            param(1) = param_alpha;
            param(2) = param_sd;
            param(3) = param_p;
            
        case 'BS2'
            param(1) = param_alpha;
            param(2) = param_sd;
            param(3) = param_p; param(3).name = 'p0';
            param(4) = param_p; param(4).name = 'p1';
            
        case {'BS2_multi'}
            param(1) = param_alpha;
            param(2) = param_sd;
            param(3) = param_p; param(3).name = 'p0';
            param(4) = param_p; param(4).name = 'p1';
            param(5) = param_p; param(5).name = 'p2';
    end
    
end