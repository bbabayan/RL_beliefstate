function y = lognormpdf(x,mu,sigma)
    
    % Log likelihood of normal distribution.
    
    y = -0.5 * ((x - mu)./sigma).^2 - log((sqrt(2*pi) .* sigma));