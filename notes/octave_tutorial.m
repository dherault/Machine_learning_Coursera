%% help and utils

help whatever
pwd
cd
ls
who % displays variables in the current scope (!!!)
whos % same but different (think ls -l)
clear % garbage everything
format short % not many decimals displayed
format long % a lot of them
fn1, fn2, fn3 % comma chainning

%% Basic operations

1+2
3-6
1/2
2^9
1 == 2 % 0 (false)
1 ~= 2 % 1 (true, not equals to)
1 && 0 % AND
1 || 0 % OR
xor(1, 0)

% To change the octave prompt

PS1('>> '); % Or whatever

%% Variables

a = 3
a = 3; % Does not print result (;)
b = 'hi!'; % single quotes
c = (3>=1);
d = pi % Math.PI

disp(d); % 3.141592 (display)
% to print strings:
disp(sprintf('2 decimals: %0.2f', d)); % 2 decimals: 3.14

%% Matrices

A = [1 2; 3 4; 5 6];
V = [1 2 3]; % 1x3 matrix (row vector)
V = [1; 2; 3] % colum vector
V = 1:0.1:2 % row vector, starts at 1, finishes at 2, increments by 0.1
V = 1:6 % 1-->6 step 1
ones(2, 3) % 2x3 matrix, only 1s
2 * ones(2, 3) % only 2s
zeroes(1, 3) % only 0s
rand(1, 3) % rand numbers matrix
randn(1, 3) % galaxy distribution (normalized)
eye(4, 4) % identity matrix
eye(4) % same

%% Moving data around

A = [1 2; 3 4; 5 6]
size(A) % a one by two matrix 
size(A, 1) % number of rows
size(A, 2) % number of columns

V = [1 2 3 4]
length(V) % 4, size of the biggest dimension

load file.ext;
load('file.ext');

V = foo(1:10) % the first 10 column of foo

save filename.ext V; % writes file in cwd, in binary format
save filename.ext V -ascii % test/ascii format

%% Manipulating data

A(3,2) % row 3, column 2 --> 6
A(2,:) % fetch everything in row 2 (: means everything)
A(:,1) % whole first column
A([1 3], :) % get everything from first and third row
A(:,2) = [10; 11; 12] %yes, column assignment
A = [A, [100; 101; 102]] % apprend another column vector to the right
A(:) % tricky! put all elements of A into a single column vector
C = [A B] % concat matrices like a god
C = [A; B] % bottom concat

%% Computing on data

A = [1 2; 3 4; 5 6];
B = [11 12; 13 14; 15 16];
C = [1 1; 2 2];
V = [1;2;3];

A*C % 3x2 matrix
A .* B % multiplication element wise
A .^ 2 % element wise squaring
1 ./ V
1 ./ A
log(V) % element wise log
exp(V) % element wise exp
abs(V) % element wise abs
-V % same as -1 * V
V + ones(length(V), 1); % 2; 3; 4
V + 1; % does the same
A' % transpose
(A')' == A
val = max(V) % 3
[val, ind] = max(V) % val = 3, ind = 3
V < 2 % [1; 0; 0] % element wise comparison

[r, c] = find(A >= 7) % find row and col that match predicate
sum, prod, floor % are cool too

max(rand(3), rand(3)) % returns a 3x3, element-wise max

max(A, [], 1) % column wise maximum
max(A, [], 2) % row wise maximum
max(max(A)) % global max
max(A(:)) % same, global max

sum(A, 1) % colum wise sum
sum(A, 2) % row wise sum
sum(sum(A .* eye(3))) % sum of diagonal values

flipud(eye(9)) % flip matrices like crepes
pinv(A) % inversion !!! pseudo inversion
inv(A) % real inversion

%% Plotting data

t = [0:0.01:0.98];
y1 = sin(2*pi*4*t);
y2 = cos(2*pi*4*t);
plot(t, y1) % hello plot
hold on; % this is a real cmd... allows 'plotting on top'
plot(t, y2, 'r')
xlabel('time')
ylabel('value')
legend('sin', 'cos')
title('my awesome plot')
print -dpng 'myplot.png' % !!!
close % closes a figure

figure(1); plot(t, y1);
figure(2); plot(t, y2); % two figures, with names

subplot(1, 2, 1); % divides plot into a 1x2 grid, acces first element
plot(t, y1);
subplot(1, 2, 2);
plot(t, y2); % two plots on the same window

axis([0.5 1 -1 1]) % changes the axes

A = magic(5);
imagesc(A); % plots a matrix as image :)
imagesc(A), colorbar, colormap gray; % BnW with a legend (colorbar)

%% Control statements

V = zeroes(10, 1);

for i=1:10
% for i = 0:n
  V(i) = 2^i;
end;
% same as
indices = 1:10 % 1 to 10 array, step 1
for i=indices
  % stuff
end;

i = 1;
while i <= 5
  v(i) = 100;
  i = i + 1;
end;

while true
  V(i) = 999;
  i = i + 1;
  if i == 6
   break;
  end;
end;

V(1) = 2;
if V(1) == 1
  disp('The value is one')
elseif V(1) == 2
  disp('The value is two')
else 
  disp('yolo')
end;

% also, exit cmd exists

% in a file:
function y = squareThisNumber(x)
y = x^2;
% end file

squareThisNumber(5) % 25

% add search path:
addpath('path')

% in a file
function [y1, y2] = squareAndCubeThisNumber(x)

y1 = x^2;
y2 = x^3;
% end file

[a, b] = squareAndCubeThisNumber(5); %a == 5, b == 125

% cost function

X = [ 1 1; 1 2 ; 1 3 ];
y = [1;2;3];

% file costFunctionJ.m:
function J = costFunctionJ(X, y, theta)

m = size(X, 1); % number of training examples
predictions = X*theta; % predictions of hypothesis on all m examples
sqrErrors = (predictions - y) .^ 2; % squarred errors

J = 1 / (2*m) * sum(sqrErrors);
% end file

%% Vectorization

% you know it (you better)

%% Normal equation noninvertibility

%{
What if a matrix is non-invertible (like X' * X)
- Redundant features (linearly dependent)
- Too many features --> delete some features, or use regularization
%}
