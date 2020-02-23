function [logp, grad] = gradrbPosterior(theta, a, b)
   x = theta(1);
   y = theta(2);
   logp = -(x^2 - 2 * x + 1 + a * y^2 - 2 * a * b * y * x^2 + a * b^2 * x^4);
   grad_x = -(2 * x - 2 - 4 * a * b * y * x + 4 * a * b^2 * x^3);
   grad_y = -(2 * a * y - 2 * a * b * x^2);
   grad = [grad_x; grad_y];
end