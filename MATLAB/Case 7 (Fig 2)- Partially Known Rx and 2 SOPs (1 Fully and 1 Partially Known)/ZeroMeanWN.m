function [wx, wy, wtr, wtrdot, wts1, wtsdot1, wts2, wtsdot2] =  ...
    ZeroMeanWN(qx, qy, S_wtr, S_wtrdot, S_wts, S_wtsdot, L)

% This function find the Zero-Mean White Processes for all processes

wx = sqrt(qx)*randn(L,1);                   % Dynamic's Zero Mean White Noise
wy = sqrt(qy)*randn(L,1);

wtr = sqrt(S_wtr)*randn(L,1);               % Rx's Clock Error Zero Mean White Noise
wtrdot = sqrt(S_wtrdot)*randn(L,1);

wts1 = sqrt(S_wts)*randn(L,1);              % SOP 1's Clock Error Zero Mean White Noise
wtsdot1 = sqrt(S_wtsdot)*randn(L,1);

wts2 = sqrt(S_wts)*randn(L,1);              % SOP 2's Clock Error Zero Mean White Noise
wtsdot2 = sqrt(S_wtsdot)*randn(L,1);

end