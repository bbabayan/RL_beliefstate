function [lik,Yhat] = myregress(X,Y)
    
    % Linear regression.
    %
    % USAGE: [lik,Yhat] = myregress(X,Y)
    %
    % INPUTS:
    %   X - design matrix
    %   Y - outcome vector
    %
    % OUTPUTS:
    %   lik - log-likelihood
    %   Yhat - predicted outcomes
    
    
    %remove missing data points for the regression(for DA data mostly)
    X2=X;
    Y2=Y;
    ind=isnan(Y2);
    Y2(ind)=[];
    X2(ind)=[];
    
    X2 = [ones(size(X2,1),1) X2];  % add intercept
    b = (X2'*X2)\(X2'*Y2);
    Yhat2 = X2*b;
    sdy = sqrt(mean((Y2-Yhat2).^2));
    lik = sum(log(normpdf(Y2,Yhat2,sdy)));
    
    %store and estimate latents for missing data points
    X = [ones(size(X,1),1) X]; 
    Yhat = X*b;