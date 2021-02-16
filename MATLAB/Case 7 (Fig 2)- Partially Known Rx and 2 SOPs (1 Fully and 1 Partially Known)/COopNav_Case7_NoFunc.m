% PAPER: Observability Analysis of Collagborative Opportunistic Navigation
% with Pseudorange Measurements
% DATE: November 3rd, 2020
% AUTHOR: Alex Nguyen
% DESCRIPTION: Replicate Figure 1 for case 4 data which involves an unknown
% reciever and two SOPs (one fully known and one paritally known)

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
[wx, wy, wtr, wtrdot, wts1, wtsdot1, wts2, wtsdot2] ...
 = ZeroMeanWN(qx, qy, S_wtr, S_wtrdot, S_wts, S_wtsdot, L); 

%----------Reciever's State Vector
rr = @(xr, yr) [xr, yr]';
rdotr = @(xrdot, yrdot) [xrdot, yrdot]';
xr1 = @(xr, yr, xrdot, yrdot, tr, trdot) [rr(xr, yr)', rdotr(xrdot, yrdot)', tr, trdot]'; % Rx's "k" State Vector
wr = @(wx, wy, wtr, wtrdot) [wx, wy, wtr, wtrdot]';                                       % Rx's Clock Errors Vector
 
Fclk = [1, T; ...
        0, 1];                              % Matrix for Clock Bias  

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

Qpv_vel = Qpv(3:4, 3:4);          % Velocity Covariance Matrix

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

%----------Partially Known Rx & Fully Known SOP 1 Dynamics
x0sop1 = [50; 100; 1; 0.1];     % Initial SOP 1 Conditions 
x0 = [0; 0; 0; 25; 10; 1];      % Initial Rx Conditions 

xsop1 = zeros(size(Fs, 1), L); % Preallocation SOP 1
wns1 = zeros(2, L); wns2 = wns1;   
xRx = zeros(size(Fr, 1), L-1); % Rx True State Preallocation 
wk = zeros(4, L);              % White Noise Preallocation
for i = 1:L
    if i == 1
        % Initial Step
        xRx(:, i) = x0; % Rx
        xsop1(:, i) = x0sop1; % SOP 2
        
    else
        % Next Step
        wns1(:, i) = ws(wts1(i), wtsdot1(i));
        xsop1(:, i) = Fs*x0sop1 + Ds*wns1(:, i);     % "k+1" SOP 1 State Vector
        
        wk(:, i) = wr(wx(i), wy(i), wtr(i), wtrdot(i));
        xRx(:, i) = Fr*x0 + Dr*wk(:, i);     % "k+1" Rx State Vector
        
        % Update
        x0sop1 = xsop1(:, i);
        x0 = xRx(:, i);
        
    end    
    
end

%----------EKF State Estimation
Fk = blkdiag(Fr(3:end, 3:end), Fs);            % State Jacobian Matrix   

f = @(x) [x(1); ...       % Rx and SOP 2 State Equations Function
          x(2); ...
          x(3) + T*x(3); ...
          x(4); ...
          x(5); ...
          x(6); ...
          x(7) + T*x(8); ...
          x(8)];

c = 299792458;                                 % Speed of Light [m/s]
n = 8;                                         % Number of States
r = 100;                                       % Observation Noise Spectral Density [m^2] 
P_est0_rx = 1e3*diag([1, 1, 30, 0.3]);         % Initial Estimation Error Covariance Matrix of Rx
P_est0_sop2 = 1e3*diag([1, 1, 30, 0.3]);       % Initial Estimation Error Covariance Matrix of SOP 2
P_est0 = blkdiag(P_est0_rx, P_est0_sop2);
Q = blkdiag(Qpv_vel, Qclk_r, Qs);
R = eye(2)*r;

xrx = [0, 25, 10, 1]';                         % Initial Rx State 
% xrx = [10, 5, 2, 20, 10, 1]';                % Initial Rx State 
xs2 = [-50, 100, 1, 0.1]';                     % Initial SOP 2 State
xsa = [xrx; xs2];                              % True Initial State Values
x_est0 = xsa + sqrt(P_est0)*rand(n,1);         % Initial Estimate State with Noise

z = zeros(2, L);                               % Preallocation
x_est = zeros(n, L); 
P_est = x_est;
x_true = x_est;

for k = 1:L
    % Measurement SOP 1 Equation Function
    h1 = @(x) sqrt((xRx(1, k) - xsop1(1, k)).^2 + (xRx(2, k) - xsop1(2, k)).^2) ...
        + c*(x(3)/c - xsop1(3, k)/c);
    
    % Measurement SOP 2 Equation Function
    h2 = @(x) sqrt((xRx(1, k) - x(5)).^2 + (xRx(2, k) - x(6)).^2) ...
        + c*(x(3)/c - x(7)/c);          

    % Observation Jacobian Matrix Function (8x2)
    Hk = @(x)  [0,  ...
                0, ...
                1, ...
                0, ...
                0, ...
                0, ...
                0, ...
                0; ...

                0, ...
                0, ...
                1, ...
                0, ...
                (-xRx(1, k) + x(5))./norm(xRx(1:2, k) - rs(x(5), x(6))), ...
                (-xRx(2, k) + x(6))./norm(xRx(1:2, k) - rs(x(5), x(6))), ...
                -1, ...
                0];                     

    % True Pseudorange Measurment SOP 1 & 2
    z(:, k) = [h1(xsa); h2(xsa)] + sqrt(r)*randn(2, 1);  

    % True State Values (Rx and SOP 2)
    x_true(:, k) = xsa;                   
    
    if k == 1
        % Predict
        x_estn = x_est0;
        P_estn = P_est0;
        
        % Update
        H = Hk(x_estn);
        yk_res = z(:, k) - [h1(x_estn); h2(x_estn)];

        P_xy = P_estn*H';
        P_yy = H*P_estn*H' + R;
        Kk = P_xy/P_yy;

%         Sk = H*P_estn*H' + r;
%         Kk = P_estn*H'*inv(Sk);
         
        % Next Step
        x_est0 = x_estn + Kk*yk_res;
        A = eye(n) - Kk*H;
        P_est0 = A*P_estn*A' + Kk*R*Kk';

%         x_est0 = x_estn + Kk*yk_res;
%         P_est0 = (eye(n) - Kk*H)*P_estn;

        xsa = f(xsa) + sqrt(Q)*randn(n, 1);
        
        % Save Estimate Values
        x_est(:, k) = x_est0;
        P_est(:, k) = diag((P_est0));
        
    else
        % Predict
        x_estn = f(x_est0);
        P_estn = Fk*P_est0*Fk' + Q;
        
        % Update
        H = Hk(x_estn);
        yk_res = z(:, k) - [h1(x_estn); h2(x_estn)];

        P_xy = P_estn*H';
        P_yy = H*P_estn*H' + R;
        Kk = P_xy/P_yy;

%         Sk = H*P_estn*H' + r;
%         Kk = P_estn*H'*inv(Sk);
        
        % Next Step
        x_est0 = x_estn + Kk*yk_res;
        A = eye(n) - Kk*H;
        P_est0 = A*P_estn*A' + Kk*R*Kk';

%         x_est0 = x_estn + Kk*yk_res;
%         P_est0 = (eye(n) - Kk*H)*P_estn;

        xsa = f(xsa) + sqrt(Q)*randn(n, 1);
        
        % Save Values
        x_est(:, k) = x_est0;
        P_est(:, k) = diag((P_est0));
    end
end

% Error Calculations
x_tilde = x_true - x_est;    % Estimation Error Trajectories
xP = sqrt(P_est);            % Estimation Error Variance Bounds

%----------Plot Results
figure; % Velocity Error Trajectory Rx
subplot(4,2,1)
x = plot(t, x_tilde(1, :), 'linewidth', 2); hold on;
plot(t, 2*xP(1, :), 'r:',  'linewidth',1.5); 
plot(t, -2*xP(1, :), 'r:', 'linewidth',1.5); hold off;
ylabel('$\dot{x}_r$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_1(t_k)$','$\pm 2 \sigma_1(t_k)$','interpreter','latex')
ax = x.Parent; set(ax, 'XTick', []); 

subplot(4,2,2)
x = plot(t, x_tilde(2, :), 'linewidth', 2); hold on;
plot(t, 2*xP(2, :), 'r:', 'linewidth',1.5); 
plot(t, -2*xP(2, :), 'r:', 'linewidth',1.5); hold off;
ylabel('$\dot{y}_r$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_2(t_k)$','$\pm 2 \sigma_2(t_k)$','interpreter','latex')
ax = x.Parent; set(ax, 'XTick', []); 

subplot(4,2,3)  % Receiver Clock Errors
x = plot(t, x_tilde(3, :), 'linewidth', 2); hold on;
plot(t, 2*xP(3, :), 'r:', 'linewidth',1.5); 
plot(t, -2*xP(3, :), 'r:', 'linewidth',1.5); hold off;
ylabel('$\tilde{c\delta t}_r$ [m/s]', 'interpreter', 'latex')
legend('$\tilde{x}_3(t_k)$','$\pm 2 \sigma_3(t_k)$','interpreter','latex')
ax = x.Parent; set(ax, 'XTick', []); set(ax, 'YTick', -20:10:20);

subplot(4,2,4)
x = plot(t, x_tilde(4, :), 'linewidth', 2); hold on;
plot(t, 2*xP(4, :), 'r:' , 'linewidth',1.5); 
plot(t, -2*xP(4, :), 'r:' , 'linewidth',1.5); hold off;
ylabel('$\tilde{c\delta \dot{t}}_r$ [m/s]', 'interpreter', 'latex')
legend('$\tilde{x}_4(t_k)$','$\pm 2 \sigma_4(t_k)$','interpreter','latex')
ax = x.Parent; set(ax, 'XTick', []); set(ax, 'YTick', -40:20:40);

subplot(4,2,5) % SOP 2 Location
x = plot(t, x_tilde(5, :), 'linewidth', 2); hold on;
plot(t, 2*xP(5, :), 'r:', 'linewidth',1.5); 
plot(t, -2*xP(5, :), 'r:', 'linewidth',1.5); hold off;
ylabel('$\tilde{x}_{s2}$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_5(t_k)$','$\pm 2 \sigma_5(t_k)$','interpreter','latex')
ax = x.Parent; set(ax, 'XTick', []); set(ax, 'YTick', -80:20:80);

subplot(4,2,6)
x = plot(t, x_tilde(6, :), 'linewidth', 2); hold on;
plot(t, 2*xP(6, :), 'r:' , 'linewidth',1.5); 
plot(t, -2*xP(6, :), 'r:' , 'linewidth',1.5); hold off;
ylabel('$\tilde{y}_{s2}$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_6(t_k)$','$\pm 2 \sigma_6(t_k)$','interpreter','latex')
ax = x.Parent; set(ax, 'XTick', []); set(ax, 'YTick', -80:20:80);

subplot(4,2,7)  % SOP 2 Clock Bias
x = plot(t, x_tilde(7, :), 'linewidth', 2); hold on;
plot(t, 2*xP(7, :), 'r:', 'linewidth',1.5); 
plot(t, -2*xP(7, :), 'r:', 'linewidth',1.5); hold off;
xlabel('time [s]'); ylabel('$\tilde{c \delta t}_{s2}$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_7(t_k)$','$\pm 2 \sigma_7(t_k)$','interpreter','latex')
ax = x.Parent; set(ax, 'XTick', 0:1:10); set(ax, 'YTick', -100:50:100);

subplot(4,2,8)
x = plot(t, x_tilde(8, :), 'linewidth', 2); hold on;
plot(t, 2*xP(8, :), 'r:' , 'linewidth',1.5); hold on;
plot(t, -2*xP(8, :), 'r:' , 'linewidth',1.5); hold off;
xlabel('time [s]'); ylabel('$\tilde{c \delta \dot{t}}_{s2}$ [m]', 'interpreter', 'latex')
legend('$\tilde{x}_8(t_k)$','$\pm 2 \sigma_8(t_k)$','interpreter','latex')
ax = x.Parent; set(ax, 'XTick', 0:1:10); set(ax, 'YTick', -40:20:40);
sgtitle('Case 7: Estimation Error Trajectories with $\pm 2\sigma$ Bounds','interpreter','latex')

for ii = 1:8
    sph = subplot(4,2,ii); % Resize Subplots
    dx0 = -0.04;
    dy0 = -0.025;
    dwithx = 0.05;
    dwithy = 0.025;
    set(sph,'position',get(sph,'position')+[dx0,dy0,dwithx,dwithy])
end

% Reciver Dynamics (Partially Known) with SOP 1 (Known) & SOP 2 (Paritally
% Known)
figure; 
plot(xRx(1,:), xRx(2,:), 'b', 'linewidth', 2); hold on;
plot(xsop1(1,:), xsop1(2,:),'bs','linewidth',5);
plot(x_est(5,:), x_est(6,:),'rs','markersize',10); hold off;
title('Case 7: System Dynamics and SOP Locations')
legend('Known Rx', 'Known SOP 1', 'Estimated SOP 2', 'location', 'best')
xlabel('x [m]'); ylabel('y [m]')
axis tight
grid on;

