% Basic operations

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

% Variables

a = 3
a = 3; % Does not print result (;)
b = 'hi!';
c = (3>=1);
d = pi % Math.PI

disp(d); % 3.141592 (display)
% to print strings:
disp(sprintf('2 decimals: %0.2f', d)); % 2 decimals: 3.14

% Matrices

A = [1 2; 3 4; 5 6];
V = [1 2 3]; % 1x3 matrix (row vector)
V = [1; 2; 3] % colum vector
V = 1:0.1:2 % row vector, starts at 1, finishes at 2, increments by 0.1
V = 1:6 % 1-->6 step 1
ones(2, 3) % 2x3 matrix, only 1s
2 * ones(2, 3) % only 2s
