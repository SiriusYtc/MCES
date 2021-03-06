%% Load the data set and define a gradient function for the posterior of a 
%  Rosenbrock. 
a = 1;
b = 0; 

%  Initialize Rosenbrock model. 
f = @(theta)gradrbPosterior(theta, a, b);
theta0 = zeros(2,1);

n_warmup = 2000;
n_mcmc_samples = 10000;
n_re = 100;


theta_mces = cell(1, n_re);
L_mces = cell(1, n_re);
ESS_MCES = zeros(2, n_re);
sample_use = 1 : 2 : n_mcmc_samples;
for i = 1 : n_re
    [theta_mces{i}, L_mces{i}] = MCES_acce(f, theta0, n_warmup, n_mcmc_samples);
    ESS_MCES(:,i) = ESS(theta_mces{i}(:, sample_use));
end
