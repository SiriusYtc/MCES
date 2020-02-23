function[thetat, pt, log0]=leapfrog(theta0, p0, K, epsilon, L, f)
[~, grad0]=f(theta0);
for i=1:L
   p0 = p0 + 1/2 * epsilon * grad0;
   theta0 = theta0 + epsilon * K * p0;
   [log0,grad0] = f(theta0);
   p0 = p0 + 1/2 * epsilon * grad0;
end
thetat=theta0;
pt=p0;
end