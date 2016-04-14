function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% disp('X'), disp(size(X));
% disp('theta'), disp(size(theta));
% disp('y'), disp(size(y));

h = X * theta;
theta_nobias = [0; theta(2:end, :)];

% disp('h'), disp(size(h));
% disp('theta_nobias'), disp(size(theta_nobias)), disp(theta_nobias);

J = (1 / (2 * m)) * (sum((h - y) .^ 2) + lambda * sum(theta_nobias .^ 2));

% disp('sum(h - y) .* X'), disp(size(sum((h - y) .* X)));

grad = (1 / m) * (sum((h - y) .* X) + lambda * theta_nobias');







% =========================================================================

grad = grad(:);

end
