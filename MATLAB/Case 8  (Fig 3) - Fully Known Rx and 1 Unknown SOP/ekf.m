function [x,P]=ekf(fstate, x, P, hmeas, z, Q, R)
% EKF   Extended Kalman Filter for nonlinear dynamic systems
% [x, P] = ekf(f,x,P,h,z,Q,R) returns state estimate, x and state covariance, P 
% for nonlinear dynamic system:
%           x_k+1 = f(x_k) + w_k
%           z_k   = h(x_k) + v_k
% where w ~ N(0,Q) meaning w is gaussian noise with covariance Q
%       v ~ N(0,R) meaning v is gaussian noise with covariance R
% Inputs:   f: function handle for f(x)
%           x: "a priori" state estimate
%           P: "a priori" estimated state covariance
%           h: fanction handle for h(x)
%           z: current measurement
%           Q: process noise covariance 
%           R: measurement noise covariance
% Output:   x: "a posteriori" state estimate
%           P: "a posteriori" state covariance
%
% Example:
%{
n=3;      %number of state
q=0.1;    %std of process 
r=0.1;    %std of measurement
Q=q^2*eye(n); % covariance of process
R=r^2;        % covariance of measurement  
f=@(x)[x(2);x(3);0.05*x(1)*(x(2)+x(3))];  % nonlinear state equations
h=@(x)x(1);                               % measurement equation
s=[0;0;1];                                % initial state
x=s+q*randn(3,1); %initial state          % initial state with noise
P = eye(n);                               % initial state covraiance
N=20;                                     % total dynamic steps
xV = zeros(n,N);          %estmate        % allocate memory
sV = zeros(n,N);          %actual
zV = zeros(1,N);
for k=1:N
  z = h(s) + r*randn;                     % measurments
  sV(:,k)= s;                             % save actual state
  zV(k)  = z;                             % save measurment
  [x, P] = ekf(f,x,P,h,z,Q,R);            % ekf 
  xV(:,k) = x;                            % save estimate
  s = f(s) + q*randn(3,1);                % update process 
end
for k=1:3                                 % plot results
  subplot(3,1,k)
  plot(1:N, sV(k,:), '-', 1:N, xV(k,:), '--')
end
%}

[x1, A] = jaccsd(fstate, x);    % Nonlinear Update and Linearization at Current State
P = A*P*A' + Q;                 % Partial Update
[z1, H] = jaccsd(hmeas, x1);    % Nonlinear Measurement and Linearization
P12 = P*H';                     % Cross Covariance
% K = P12*inv(H*P12+R);         % Kalman Filter Gain
% x = x1+K*(z-z1);              % State Estimate
% P = P-K*P12';                 % State Covariance Matrix
R = chol(H*P12+R);              % Cholesky Factorization
U = P12/R;                      % K=U/R'; Faster because of Back Substitution
x = x1 + U*(R'\(z - z1));       % Back Substitution to get State Update
P = P - U*U';                   % Covariance Update, U*U'=P12/R/R'*P12'=K*P12.

end