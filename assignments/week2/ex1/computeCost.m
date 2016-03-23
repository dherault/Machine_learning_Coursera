function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);

a = [1, 2, 3];
fprintf('yolo\n');
fprintf(X);

% J = (0.5 / m) * sum

% =========================================================================

end
