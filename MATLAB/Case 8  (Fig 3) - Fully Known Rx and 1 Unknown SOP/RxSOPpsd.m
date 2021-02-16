function [S_wtr, S_wtrdot, S_wts, S_wtsdot] = ...
    RxSOPpsd(h0_rx, h_neg2_rx, h0_sop, h_neg2_sop)

% This function caclulates the PSD for both the Rx and SOP

S_wtr = h0_rx/2; S_wtrdot = 2*pi^2*h_neg2_rx;     % Rx's Power Spectral Density     
S_wts = h0_sop/2; S_wtsdot = 2*pi^2*h_neg2_sop;   % SOP's Power Spectral Density  

end