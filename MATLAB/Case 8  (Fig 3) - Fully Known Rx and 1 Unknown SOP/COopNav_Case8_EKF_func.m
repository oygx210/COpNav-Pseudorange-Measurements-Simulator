% PAPER: Observability Analysis of Collagborative Opportunistic Navigation
% with Pseudorange Measurements
% DATE: November 2nd, 2020
% AUTHOR: Alex Nguyen
% DESCRIPTION: Replicate Figure 3 for case 8 data which involves one fully
% known Rx and one unknown SOP

clc; clear; close all;

%-----------Simulation Time
T = 10e-3;                                  % Sampling Period [s]
t = [0:T:10]';                                 % Experiment Time Duration [s]

%----------Power Spectral Density
h0_rx = 2e-19; h_neg2_rx = 2e-20;           % Rx's Clocks
h0_sop = 8e-20; h_neg2_sop = 4e-23;         % SOP's Clocks 
[S_wtr, S_wtrdot, S_wts, S_wtsdot] = RxSOPpsd(h0_rx, h_neg2_rx, h0_sop, h_neg2_sop);

%----------Rx and SOP: Zero-Mean White Noise Processes
L = length(t);                              % Sample length for the random signal
qx = 0.1; qy = qx;                          % Rx's Process Noise Spectral Density [m^2/s^4] (wr covariance)
[wx, wy, wtr, wtrdot, wts, wtsdot] = ZeroMeanWN(qx, qy, S_wtr, S_wtrdot, S_wts, S_wtsdot, L); 

%----------Rx's State Vector
rr = @(xr, yr) [xr, yr]';
rdotr = @(xrdot, yrdot) [xrdot, yrdot]';
xr1 = @(xr, yr, xrdot, yrdot, tr, trdot) [rr(xr, yr)', rdotr(xrdot, yrdot)', tr, trdot]'; % Rx's "k" State Vector
wr = @(wx, wy, wtr, wtrdot) [wx, wy, wtr, wtrdot]';                                       % Rx's Clock Errors Vector
 
Fclk = [1, T; ...
        0, 1];                              % Matrix within Block Matrix Fr 

Fr = [eye(2), T*eye(2), zeros(2); ...
      zeros(2), eye(2), zeros(2); ...
      zeros(2), zeros(2), Fclk];            % Matrix Multiplied to xr(tk)  
  
Dr = [zeros(2,4); eye(4)];                  % Matrix Multiplied to wr(tk)

%----------Rx's Covariance Matrices
Qclk_r = [S_wtr*T + S_wtrdot*T^3/3, S_wtrdot*T^2/2; ...
          S_wtrdot*T^2/2, S_wtrdot*T];

Qpv = [qx*T^3/3, 0, qx*T^2/2, 0; ...
       0, qy*T^3/3, 0, qy*T^2/2; ...
       qx*T^2/2, 0, qx*T, 0; ...
       0, qy*T^2/2, 0, qy*T];

Qr = blkdiag(Qpv, Qclk_r);      % DT zero-mean white noise sequence Rx covariance   

%----------SOP's State Vector
rs = @(xs, ys) [xs, ys]';
xs1 = @(xs, ys, ts, tsdot) [rs(xs, ys)', ts, tsdot]'; % SOP's "k" State Vector
ws = @(wts, wtsdot) [wts, wtsdot]';                   % SOP's Clock Errors

Fs = blkdiag(eye(2), Fclk);                           % Matrix Multiplied to xs(tk)
  
Ds = [zeros(2); eye(2)];                              % Matrix Multiplied to ws(tk)

%----------SOP's Covariance Matrices
Qclk_s = [S_wts*T + S_wtsdot*T^3/3, S_wtsdot*T^2/2; ...
          S_wtsdot*T^2/2, S_wtsdot*T];
    
Qs = blkdiag(zeros(2), Qclk_s); % DT zero-mean white noise sequence SOP covariance

%----------Find Reciever's Dynamics
x0 = [0; 0; 0; 25; 10; 1];     % Initial Rx Conditions 
xRx = zeros(size(Fr, 1), L-1); % Rx True State Preallocation 
wk = zeros(4, L);              % White Noise Preallocation
for i = 1:L
    if i == 1
        % Initial Step
        xRx(:, i) = x0;
        
    else
        % Next Step
        wk(:, i) = wr(wx(i), wy(i), wtr(i), wtrdot(i));
        xRx(:, i) = Fr*x0 + Dr*wk(:, i);     % "k+1" Rx State Vector
        
        % Update
        x0 = xRx(:, i);
        
    end    
    
end

%- ---------EKF State Estimation 
c = 299792458;% Speed of Light [m/s]
N = L;        % Total Dynamic Steps
n = 4;        % Number of States
r = 10;       % Measurement Standard Deviation
R = r^2;      % Measurement Noise Covariance 
Q = Qs;       % Process Noise Covariance 
q = sqrt(Qs); % Process Noise Standard Deviation 
  
f = @(x) [x(1); ...             % Nonlinear States Equation Function
          x(2); ...
          x(3) + T*x(4); ...
          x(4)];  
      
P = 1e3*diag([1, 1, 30, 0.3]);  % Initial Estiation Error Covariance Matrix 
s = [50; 100; 1; 0.1];          % Initial SOP States
x = s + q*randn(n,1);           % Initial SOP State with Noise
x0 = [0; 0; 0; 25; ...
    10; 1; 50; 100; 1; 0.1];    % Initial Reciever Conditions 
                 
xV = zeros(n,N);          % Estimate SOP State Preallocation    
sV = zeros(n,N);          % Actual SOP State Preallocation
zV = zeros(1,N);          % Measurement Preallocation
xP = zeros(4,N);          % Covariance Preallocation

for k = 1:N
    % Measurement Equation Function
    h=@(x) sqrt((xRx(1, k) - x(1)).^2 +(xRx(2, k) -  ...
        x(2)).^2) + c*(xRx(5, k)/c - x(3)/c);                               
    
    z = h(s) + r*randn;                     % Pseudorange Measurments
    
    % Save
    sV(:,k)= s;                             % Actual SOP State
    zV(k)  = z;                             % Pseudorange Measurments
    
    % Perform EKF
    [x, P] = ekf(f, x, P, h, z, Q, R);            
    
    % Save
    xV(:,k) = x;                            % SOP State Estimate
    xP(:,k) = sqrt(diag(P));                % Estimate Error Standard Deviation 
    
    s = f(s) + q*randn(n,1);                % Update Process
end

xtild = sV - xV;                            % Estimate Error Trajectories

%----------Plot Results
figure; % Trajectory Error Dynamics
subplot(2,1,1)
plot(t, xtild(1, :), 'linewidth',1.5); hold on;
plot(t, 2*xP(1, :), 'r',  'linewidth',1); hold on;
plot(t, -2*xP(1, :), 'r', 'linewidth',1)
xlabel('t [s]'); ylabel('$x_s$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_1(t_k)$','$\pm 2 \sigma(t_k)$','interpreter','latex')

subplot(2,1,2)
plot(t, xtild(2, :), 'linewidth',1.5); hold on;
plot(t, 2*xP(2, :), 'r', 'linewidth',1); hold on;
plot(t, -2*xP(2, :), 'r', 'linewidth',1)
xlabel('t [s]'); ylabel('$y_s$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_2(t_k)$','$\pm 2 \sigma(t_k)$','interpreter','latex')
sgtitle('Position Estimation Error for SOP', 'interpreter', 'latex')

figure; % Clock Error Estimates
subplot(2,1,2)
plot(t, xtild(3, :), 'linewidth',1.5); hold on;
plot(t, 2*xP(3, :), 'r', 'linewidth',1); hold on;
plot(t, -2*xP(3, :), 'r', 'linewidth',1)
xlabel('t [s]'); ylabel('$c \delta_{t_s}$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_3(t_k)$','$\pm 2 \sigma(t_k)$','interpreter','latex')

subplot(2,1,1)
plot(t, xtild(4, :), 'linewidth',1.5); hold on;
plot(t, 2*xP(4, :), 'r' , 'linewidth',1); hold on;
plot(t, -2*xP(4, :), 'r' , 'linewidth',1)
xlabel('t [s]'); ylabel('$c \delta_{\dot{t}_s}$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_4(t_k)$','$\pm 2 \sigma(t_k)$','interpreter','latex')
sgtitle('Clock Bias Estimation Error for SOP', 'interpreter', 'latex')

figure; % Rx Dynamics
plot(xRx(1,:), xRx(2,:), '--', 'linewidth', 2); hold on;
plot(sV(1,:), sV(2,:),'s','markersize',10)
title('True Reciever Dynamics')
legend('True Reciever Location', 'True SOP Location', 'location', 'northeast')
xlabel('x [m]'); ylabel('y [m]')
axis equal
grid on; shg;
