data = load('ex1data1.txt');

y = data(:, 2);
m = length(y); % number of training examples
global X = [ones(m, 1), data(:,1)];
global theta = zeros(2, 1);

% function hi = avg (i)
%   hi = dot(theta, X(i, :));
% endfunction
function hi = compute_h_at_i (i)
  global theta X;
  fprintf('%f\n', X);
  hi = dot(theta, X(i, :));
endfunction



% fprintf('%f\n', sum(hi([1:5])));
fprintf('theta: %f\n', theta);
fprintf('%f\n', X(5, :));
fprintf('dot: %f\n', compute_h_at_i(4));
