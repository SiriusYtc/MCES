function [theta, acc] = hmc( theta0, f, L, epsilon, K)
n_dimension = length(theta0);
if nargin < 5
    K = eye(n_dimension);
end
M = inv(K);
acc=0;

mu = zeros(n_dimension, 1);
p0 = mvnrnd(mu, M)';
[theta_star, p_star] = leapfrog(theta0, p0, K, epsilon, L, f);

U_0 = -f(theta0);
U_star = -f(theta_star);
K_0 = 1/2 * p0' * K * p0;
K_star = 1/2 * p_star' * K * p_star;
Total_Energy_0 = U_0 + K_0;
Total_Energy_star = U_star + K_star;

probability_0 = exp(-Total_Energy_0);
probability_star = exp(-Total_Energy_star);

p_threshold = min(1, probability_star/probability_0);
if isnan(probability_star)
   p_threshold = 0;
end
u = rand;
if u < p_threshold
   theta0 = theta_star;
   acc = 1;
end
theta = theta0;
end