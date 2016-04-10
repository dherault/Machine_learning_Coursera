function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% disp('y'), disp(size(y))
% disp('X'), disp(size(X))

X_bias = [ones(m, 1), X];

% disp('X_bias'), disp(size(X_bias))

a2 = sigmoid(X_bias * Theta1');

% disp('a2'), disp(size(a2))

a2_bias = [ones(size(a2, 1), 1), a2];

% disp('a2_bias'), disp(size(a2_bias))

h = sigmoid(a2_bias * Theta2');

% disp('h'), disp(size(h))
% fprintf('\nWill now loop through examples.\n\n')

recoding_vector = [1:max(y)];

% disp('recoding_vector (exact)'), disp(recoding_vector)

for i=1:m
  
  recoded_y = y(i, 1) ==  recoding_vector;
  subsetof_h = h(i, :);
  local_J = -recoded_y .* log(subsetof_h) - (1 - recoded_y) .* log(1 - subsetof_h);
  
  J = J + sum(local_J);
  
  if (i == 1) 
    % disp('recoded_y'), disp(size(recoded_y));
    % disp('subsetof_h'), disp(size(subsetof_h));
    % disp('local_J'), disp(size(local_J));
  end
end

J = J / m; % non-regularized ok :)

% fprintf('\nWill now apply regularization.\n\n')

regularized_suffix = 0;
n_layers = 3; 

% disp('Theta1'), disp(size(Theta1))
% disp('Theta2'), disp(size(Theta2))

Theta1_nobias = Theta1(:, 2:end);
Theta2_nobias = Theta2(:, 2:end);

% disp('Theta1_nobias'), disp(size(Theta1_nobias))
% disp('Theta2_nobias'), disp(size(Theta2_nobias))

% hardcoded number of layers == being soft!
% damn I can't find how to create a list of matrices

[s1_row, s1_col] = size(Theta1_nobias);
[s2_row, s2_col] = size(Theta2_nobias);

for i=1:s1_row
  for k=1:s1_col
    regularized_suffix = regularized_suffix + Theta1_nobias(i, k) ^ 2;
  end
end
for i=1:s2_row
  for k=1:s2_col
    regularized_suffix = regularized_suffix + Theta2_nobias(i, k) ^ 2;
  end
end

J = J + lambda / (2 * m) * regularized_suffix; % regularized ok :)

% Backpropagation

% disp('')
% disp('!/__Backpropagation__\!')
% disp('')

% delta1 = zeros(size(Theta1_nobias));
% delta2 = zeros(size(Theta2_nobias));
% disp('delta1'), disp(size(delta1));
% disp('delta2'), disp(size(delta2));

for t=1:m
  a1_bias = [1, X(t,:)];
  z2 = a1_bias * Theta1';
  a2_bias = [1, sigmoid(z2)];
  a3 = sigmoid(a2_bias * Theta2');
  
  d3 = a3 - (y(t, 1) ==  recoding_vector);
  d2 = d3 * Theta2_nobias .* sigmoidGradient(z2);
  
  Theta1_grad = Theta1_grad + d2' * a1_bias;
  Theta2_grad = Theta2_grad + d3' * a2_bias;
  
  if (t == 1) 
    % disp('[1, a1]'), disp(size([1, a1]));
    % disp('d2t'), disp(size(d2' * [1, a1]));
    % disp('a3'), disp(size(a3));
    % disp('d3'), disp(size(d3));
    % disp('Theta1_nobias'), disp(size(Theta1_nobias));
    % disp('Theta2_nobias'), disp(size(Theta2_nobias));
    % disp('sigmoidGradient(z2)'), disp(size(sigmoidGradient(z2)));
    % disp('d2'), disp(size(d2));
    % disp('a1t * d2'), disp(size(d2' * a1));
    % disp('d3'), disp(size(d3));
    % disp('a2'), disp(size(a2));
  end
end

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% disp('')
% disp('!\__Backpropagation__/!')
% disp('')

% disp('')
% disp('!/__Regularization__\!')
% disp('')

% disp('Theta1_grad'), disp(size(Theta1_grad));
% disp('Theta2_grad'), disp(size(Theta2_grad));

Theta1_grad_nobias_regularized = Theta1_grad(:, 2:end) + (lambda / m) * Theta1_nobias;
Theta2_grad_nobias_regularized = Theta2_grad(:, 2:end) + (lambda / m) * Theta2_nobias;

Theta1_grad = [Theta1_grad(:, 1), Theta1_grad_nobias_regularized];
Theta2_grad = [Theta2_grad(:, 1), Theta2_grad_nobias_regularized];

% disp('Theta1_grad_nobias_regularized'), disp(size(Theta1_grad_nobias_regularized));
% disp('Theta2_grad_nobias_regularized'), disp(size(Theta2_grad_nobias_regularized));

% disp('')
% disp('!\__Regularization__/!')
% disp('')

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
