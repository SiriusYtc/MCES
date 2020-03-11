%% Load the data set and define a gradient function for the posterior of a  
%  german credit data
load('german.mat', 'y', 'X')
[n, d] = size(X);

% Initialize Bayesian logistic regression model.
X = (X - repmat(mean(X,1), n, 1)) ./ repmat(std(X,[],1), n, 1);
X = [ones(n,1), X];
f = @(beta) gradLogitPosterior(beta, y, X);
theta0 = zeros(d+1, 1);


n_warmup = 2000;
n_mcmc_samples = 10000;
n_re = 100; 

theta_mces = cell(1, n_re);
L_mces = cell(1, n_re);
ESS_MCES = zeros(d+1, n_re);
sample_use = 1 : 2 : n_mcmc_samples;
for i = 1 : n_re
    [theta_mces{i}, L_mces{i}] = MCES_acce(f, theta0, n_warmup, n_mcmc_samples);
    ESS_MCES(:,i) = ESS(theta_mces{i}(:, sample_use));
end
