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

%----------Reciever's State Vector
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

%----------Reciever's Covariance Matrices
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

%----------Known Reciever Dynamics
x0 = [0; 30; 2; 25; 10; 0.1];     % Initial Rx Conditions 
xRx = zeros(size(Fr, 1), L);   % Rx True State Preallocation 
wk = zeros(4, L);              % White Noise Preallocation
for i = 1:L
    if i == 1           % Initial Step
        
        xRx(:, i) = x0; 
        
    else                % Next Step
        
        wk(:, i) = wr(wx(i), wy(i), wtr(i), wtrdot(i));
        xRx(:, i) = Fr*x0 + Dr*wk(:, i);     % "k+1" Rx State Vector
        
        % Update
        x0 = xRx(:, i);
    end
end

%----------EKF State Estimation
Fk = [1 0 0 0; ...
      0 1 0 0; ...        % State Jacobian Matrix
      0 0 1 T; ...
      0 0 0 1];    

f = @(x) [x(1); ...       % SOP State Equations Function
    x(2); ...
    x(3) + T*x(4); ...
    x(4)];

n = 4;                                         % Number of States
c = 299792458;                                 % Speed of Light [m/s]
r = 100;                                       % Observation Noise Spectral Density [m^2] 
Ps_est0 = 1e3*diag([1, 1, 30, 0.3]);           % Initial Estimation Error Covariance Matrix of SOP 1
xsa = [25, 150, 1, 0.1]';                      % Initial SOP 1 State 
xs_est0 = xsa + sqrt(Ps_est0)*rand(n, 1);           % Inital SOP 1 State with Noise

z = zeros(L, 1);                               % Preallocation
xs_est = zeros(n, L); 
Ps_est = xs_est;
xs_true = xs_est;
for k = 1:L
    % Measurement Equation Function
    h = @(x) norm(xRx(1:2, k) - rs(x(1), x(2))) ...
        + c*(xRx(5, k)/c - x(3)/c);
    
    % Observation Jacobian Matrix Function (nx1)
    Hk = @(x) [(-xRx(1, k) + x(1))./norm(xRx(1:2, k) - rs(x(1), x(2))), ...
               (-xRx(2, k) + x(2))./norm(xRx(1:2, k) - rs(x(1), x(2))), ...
                -1, ...
                 0];
    
    z(k) = h(xsa) + sqrt(r)*randn;          % True Pseudorange Measurment
    xs_true(:, k) = xsa;                    % True SOP Values
    
    if k == 1
        % Predict
        xs_estn = xs_est0;
        Ps_estn = Ps_est0;
        
        % Update
        H = Hk(xs_estn);
        yk_res = z(k) - h(xs_estn);
        Sk = H*Ps_estn*H' + r;
        Kk = Ps_estn*H'*inv(Sk);
        
        % Next Step
        xs_est0 = xs_estn + Kk*yk_res;
        Ps_est0 = (eye(n) - Kk*H)*Ps_estn;
        xsa = f(xsa) + sqrt(Qs)*randn(n, 1);
        
        % Save Estimate Values
        xs_est(:, k) = xs_est0;
        Ps_est(:, k) = diag((Ps_est0));
        
    else
        % Predict
        xs_estn = Fk*xs_est0;
        Ps_estn = Fk*Ps_est0*Fk' + Qs;
        
        % Update
        H = Hk(xs_estn);
        yk_res = z(k) - h(xs_estn);
        Sk = H*Ps_estn*H' + r;
        Kk = Ps_estn*H'*inv(Sk);
        
        % Next Step
        xs_est0 = xs_estn + Kk*yk_res;
        Ps_est0 = (eye(n) - Kk*H)*Ps_estn;
        xsa = f(xsa) + sqrt(Qs)*randn(n, 1);
        
        % Save Values
        xs_est(:, k) = xs_est0;
        Ps_est(:, k) = diag((Ps_est0));
    end
end

xtild = xs_true - xs_est;  % Estimation Error Trajectories
xP = sqrt(Ps_est);         % Estimation Error Variance Bounds

%----------Plot Results
figure; % Trajectory Error Dynamics
subplot(2,1,1)
plot(t, xtild(1, :), 'linewidth', 2); hold on;
plot(t, 2*xP(1, :), 'r:',  'linewidth',1.5); hold on;
plot(t, -2*xP(1, :), 'r:', 'linewidth',1.5)
xlabel('t [s]'); ylabel('$x_s$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_1(t_k)$','$\pm 2 \sigma_1(t_k)$','interpreter','latex')

subplot(2,1,2)
plot(t, xtild(2, :), 'linewidth', 2); hold on;
plot(t, 2*xP(2, :), 'r:', 'linewidth',1.5); hold on;
plot(t, -2*xP(2, :), 'r:', 'linewidth',1.5)
xlabel('t [s]'); ylabel('$y_s$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_2(t_k)$','$\pm 2 \sigma_2(t_k)$','interpreter','latex')
sgtitle('Position Estimation Error for SOP', 'interpreter', 'latex')

figure; % Clock Error Estimates
subplot(2,1,1)
plot(t, xtild(3, :), 'linewidth', 2); hold on;
plot(t, 2*xP(3, :), 'r:', 'linewidth',1.5); hold on;
plot(t, -2*xP(3, :), 'r:', 'linewidth',1.5)
xlabel('t [s]'); ylabel('$c \delta_{t_s}$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_3(t_k)$','$\pm 2 \sigma_3(t_k)$','interpreter','latex')

subplot(2,1,2)
plot(t, xtild(4, :), 'linewidth', 2); hold on;
plot(t, 2*xP(4, :), 'r:' , 'linewidth',1.5); hold on;
plot(t, -2*xP(4, :), 'r:' , 'linewidth',1.5)
xlabel('t [s]'); ylabel('$c \delta_{\dot{t}_s}$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_4(t_k)$','$\pm 2 \sigma_4(t_k)$','interpreter','latex')
sgtitle('Clock Bias Estimation Error for SOP', 'interpreter', 'latex')

figure; % Rx Dynamics
plot(xRx(1,:), xRx(2,:), 'm', 'linewidth', 2.5); hold on;
plot(xs_true(1,:), xs_true(2,:),'sk','linewidth',5)
title('Known Reciever Dynamics')
legend('True Reciever Location', 'True SOP Location', 'location', 'northeast')
xlabel('x [m]'); ylabel('y [m]')
axis equal
grid on;