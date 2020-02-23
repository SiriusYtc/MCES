function[theta_MCES, L_MCES]=MCES_acce(f, theta0, n_warmup, n_mcmc, epsilon0, L0, rou, n_update, n_L, Acc_min, I_max)
% Args:
% f - function handle: returns the log probability of the target distribution
%and its gradient.
% theta0 - column vector: current state of a Markov chain and the
%     initial state for the trajectory of Hamiltonian dynamcis.
% n_warmup - int: the number of the samples sampled during the burn-in
%     peroid
% n_mcmc - int: the length of the Hamiltonian Monte-carlo chain.
% epsilon0 - double: the leapfrog parameter used during burn-in period
% L0 - int: the leapfrog parameter used during burn-in period
% rou - double : the multiplier used to update L.
% n_update - int : the maximum iteration updating the covariance M and L
% n_L - the number of samples used to calculate the acceptance probability
% Acc_min - double: the minimum of the accepatance probability need to reach
% I_max - int: the parameter used to stop the update of L.

% Returns:
% theta_MCES - matrix: the Hamiltonain Monte-Carlo chain sampled by MCES.
% L_MCES - row vector: the number of leapfrog steps for each sample in the
%     Hamiltonian markov chain


if nargin < 11
    I_max = 1;
end

if nargin < 10
    Acc_min = 0.6;
end
if nargin < 9
    n_L = ceil(n_mcmc / 50);
end

if nargin < 8
    n_update = ceil(n_mcmc / 5);
end

if nargin < 7
    rou = 1.2;
end

if nargin < 5
    epsilon0 = 0.01;
    L0 = 100;
end

n_dimension = length(theta0);
theta_warmup = zeros(n_warmup, n_dimension);
theta = theta0;
%warmup and initialize covariance    
for i = 1:n_warmup
    [theta] = hmc(theta, f, L0, epsilon0);
    theta_warmup(i , :) = theta';
end
mu = mean(theta_warmup)';
K = cov(theta_warmup);
T = pi/2;

L = 1;
alpha = 1;
acc_old = -inf;
L_old = L;
L_max = 60;
I_count = 0;


acc_sample = zeros(1,n_mcmc);
L_MCES = zeros(1, n_mcmc);
theta_MCES = zeros(n_dimension,n_mcmc);
L_count=zeros(I_max,1);
for i = 1:n_mcmc
    
    n_cal = n_warmup + 1;
    epsilon = T / L;
    [theta, acc] = hmc( theta, f, L, epsilon, K);
    
    %update covariance
    if i <= n_update
        K=K+((theta-mu)*(theta-mu)'-K)/(n_cal+1);
        mu=mu+(theta-mu)/(n_cal+1);
    end
    acc_sample(i)=acc;
    theta_MCES(:, i) = theta;
    L_MCES(i) = L;
    
    %update L
    if (mod(i, n_L) == 0) && (alpha == 1) && (i <= n_update)
       acc_new = sum(acc_sample(1, i - n_L + 1:i));
       L_new = L_MCES(i - 1);
       Acc = sum(acc_sample(i - n_L + 1:i)) / n_L;
       if Acc > Acc_min
           if acc_new / L_new < acc_old / L_old
               I_count = I_count + 1;
               if (I_count >= I_max)
                   alpha = 0;
                   L_count(I_count) = L_old;
               end
           else
               acc_old=acc_new;
               L_old=L_new;
           end
       else 
           I_count=0;
           acc_old=acc_new;
           L_old=L_new;
       end
       L = min(ceil(rou * L), L_max);
       if L == L_max
           alpha = 0;
           L_count(1) = L_max;
       end
       if alpha == 0
           if L_count(1)==0
               L_count(1)=L_new;
           end
           L = L_count(1);
       end 
    end
end