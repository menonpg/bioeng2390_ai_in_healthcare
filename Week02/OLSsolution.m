% Create x values between 0 and 1
x = linspace(0, 1, 100);  % Creates 100 points between 0 and 1 . 
% Lets say this is the original signal

% Define parameters
b1 = 5;    % Slope =  1/sd
b0 = 10;   % Y-intercept = -mu/sd

% Calculate y values
y = b1 * x + b0 + 0.8.*rand([1,100]);  % normalized value 'measurement' with some error 

% Plot the results
plot(x, y, 'r*')
xlabel('x')
ylabel('y')
title('Linear Function: y = 5x + 10')
grid on

%% Lets fit the b1 and b0 based on the measured y and the actual x
y = y';
A = [x', ones([100,1])]

solution = inv(A'*A)*(A'*y)
%   10.3051 = estimated b0
%    4.9177 = estimated b1

solutionInternal = A\y