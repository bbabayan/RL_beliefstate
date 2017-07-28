function [lik,latents] = likfun(x,data,model)

% Compute log-likelihood for a single subject. Licking is modeled as a linear function of value at the time of cue.
%
% USAGE: [lik,latents] = likfun(x,data,model)
%
% INPUTS:
%   x - parameter vector (see below for interpretation of parameters)
%   data - data structure (see fit_models.m)
%   model - which model to fit
%
% OUTPUTS:
%   lik - log-likelihood of the data
%   latents - structure with the following fields:
%       .V - [N x 1] reward expectation at time of cue
%       .rpe - [N x 1] reward prediction error at time of outcome
%       .lick - [N x 1] predicted lick signal
%
% Sam Gershman, Benedicte Babayan Jan 2017

% pre-allocate arrays
V = zeros(data.N,1);
rpe = zeros(data.N,1);

switch model
    case 'noBS0'
        % parameters
        alpha = x(1);        % learning rate
        
        for n = 1:data.N
            % reset values and prior for first trial
            if (data.trial(n)==1)
                v = 0.5;            % initial values for each state
            end
            
            V(n) = v;
            rpe(n) = data.r(n) - v;     % reward prediction error
            v = v + alpha*rpe(n);    % update values
        end
        
    case 'noBS2'
        % parameters
        alpha = x(1);        % learning rate
        v0 = x(2);           % prior following small block
        v1 = x(3);           % prior following big block
        
        for n = 1:data.N
            % reset values and prior for first trial
            if (data.trial(n)==1) && (data.previous(n)==0) % small previous block
                v = v0;            % initial values for each state
            elseif (data.trial(n)==1) && (data.previous(n)==1) % large previous block
                v = v1;            % initial values for each state
            end
            
            V(n) = v;
            rpe(n) = data.r(n) - v;     % reward prediction error
            v = v + alpha*rpe(n);    % update values
        end
        
    case 'BS0'
        
        % parameters
        alpha = x(1);       % learning rate
        sd = x(2);          % internal noise standard deviation
        
        for n = 1:data.N
            if data.trial(n)==1
                P = [0.5 0.5];
                v = data.v0;
            end
            
            V(n) = v*P';                    % value
            rpe(n) = data.r(n) - V(n);      % reward prediction error
            logp = log(P) + lognormpdf(data.r(n),v,sd);
            v = v + alpha*rpe(n)*P;         % update values
            P = exp(logp - logsumexp(logp));
        end
        
    case 'BS1'
        % parameters
        alpha = x(1);       % learning rate
        sd = x(2);          % internal noise standard deviation
        p = x(3);          % prior following small block
        
        for n = 1:data.N
            % reset values and prior for first trial
            if (data.trial(n)==1) && (data.previous(n)==0) % small previous block
                P = [p 1-p];      % initial posterior (i.e., the prior) (influence of the previous block)
                v = data.v0;        % initial values for each state
            elseif (data.trial(n)==1) && (data.previous(n)==1) % large previous block
                P = [1-p p];      % initial posterior (i.e., the prior)
                v = data.v0;        % initial values for each state
            end
            
            V(n) = v*P';                    % value
            rpe(n) = data.r(n) - V(n);      % reward prediction error
            logp = log(P) + lognormpdf(data.r(n),v,sd);
            v = v + alpha*rpe(n)*P;         % update values
            P = exp(logp - logsumexp(logp));
        end
        
    case 'BS2'
        % parameters
        alpha = x(1);       % learning rate
        sd = x(2);          % internal noise standard deviation
        p0 = x(3);          % prior following small block
        p1 = x(4);          % prior following big block
        
        for n = 1:data.N
            % reset values and prior for first trial
            if (data.trial(n)==1) && (data.previous(n)==0) % small previous block
                P = [p0 1-p0];      % initial posterior (i.e., the prior) (influence of the previous block)
                v = data.v0;        % initial values for each state
            elseif (data.trial(n)==1) && (data.previous(n)==1) % large previous block
                P = [p1 1-p1];      % initial posterior (i.e., the prior)
                v = data.v0;        % initial values for each state
            end
            
            V(n) = v*P';                    % value
            rpe(n) = data.r(n) - V(n);      % reward prediction error
            logp = log(P) + lognormpdf(data.r(n),v,sd);
            v = v + alpha*rpe(n)*P;         % update values
            P = exp(logp - logsumexp(logp));
        end
        
    case 'BS2_multi'
        % parameters
        alpha = x(1);       % learning rate
        sd = x(2);          % internal noise standard deviation
        p0 = x(3);          % prior following small block
        p1 = x(4);          % prior following big block
        p2 = x(5);          % prior for intermediate states
        u = unique(data.r); K = length(u);
        data.v0 = u';
        
        for n = 1:data.N
            % reset values and prior for first trial
            if (data.trial(n)==1) && (data.previous(n)==0) % small previous block
                P = zeros(1,K)+p2;      % initial posterior (i.e., the prior) (influence of the previous block)
                P(1) = p0; P(end) = p1; % shouldn'it it be P(end) = 1-p0 ?
                P = P./sum(P);
                v = data.v0;        % initial values for each state
            elseif (data.trial(n)==1) && (data.previous(n)==1) % large previous block
                P = zeros(1,K)+p2;      % initial posterior (i.e., the prior) (influence of the previous block)
                P(1) = p1; P(end) = p0;
                P = P./sum(P);
                v = data.v0;        % initial values for each state
            end
            
            V(n) = v*P';                    % value
            rpe(n) = data.r(n) - V(n);      % reward prediction error
            logp = log(P) + lognormpdf(data.r(n),v,sd);
            v = v + alpha*rpe(n)*P;         % update values
            P = exp(logp - logsumexp(logp));
        end
        
end

% save latents
latents.rpe = rpe;
latents.V = V;

% fit regression coefficients and observation noise, compute log-likelihood
[lik,latents.DA_US] = myregress(latents.rpe,data.DA_US);