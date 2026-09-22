function [result, fig] = tune_pi_kp(varargin)
%TUNE_PI_KP Tune and plot the PI-controller proportional gain.
    [result, fig] = tune_controller_gain(paper_gain_spec('pi_kp'), varargin{:});
end
