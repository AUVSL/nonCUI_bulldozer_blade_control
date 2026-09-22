function [result, fig] = tune_p_kp(varargin)
%TUNE_P_KP Tune and plot the P-controller proportional gain.
    [result, fig] = tune_controller_gain(paper_gain_spec('p_kp'), varargin{:});
end
