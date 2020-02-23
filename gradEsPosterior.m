function [logp, grad] = gradEsPosterior(theta0, y, sigma, mu_min, mu_max, tau_min, tau_max, prior)
% function [grad, logp] = gradLogitPosterior(beta, y, X, prior)
%
% Computes the log posterior probability of the regression coefficient
% 'beta' and its gradient for a Bayesian logistic regression model.
%
% Args:
% beta - column vector of length d+1 
% y - boolean vector (outcome variable)
% X - n by (d+1) matrix with an intercept at column index 1 (design matrix)
% prior - function handle to return log probability of the prior on 'beta'
%     (up to an additive constant) and its gradient
if nargin < 8
    % Default flat prior.
    prior = @(theta) deal(zeros(length(theta),1), 0);
end
if nargin <6
   tau_min = 0;
   tau_max = Inf;
end
if nargin <4
   mu_min = -Inf;
   mu_max = Inf;
end
theta = theta0(1:8, 1);
mu = theta0(9);
tau = theta0(10);
if (tau < tau_max) && (tau > tau_min) && (mu < mu_max) && (mu > mu_min) 
    
    [grad_hyperprior, logp_hyperprior] = prior(theta0);

    logp_prior = -sum(1/2 * (theta - mu).^ 2/ tau^ 2 + log(tau));
    logp_likelihood = -sum(1/2 * (y - theta).^ 2 ./ sigma.^ 2);

    logp = logp_hyperprior + logp_prior + logp_likelihood;

    grad_theta = -(theta-y)./ sigma.^2 - (theta - mu)/ tau^2;
    grad_mu = -sum(mu - theta)/ tau^2;
    grad_tau = -8/tau + sum((mu-theta).^2) / tau^3;

    grad=[grad_theta; grad_mu; grad_tau] + grad_hyperprior;
else
    grad=zeros(length(theta0),1);
    logp=-Inf;
end
end