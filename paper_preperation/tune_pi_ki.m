function [result, fig] = tune_pi_ki(varargin)
%TUNE_PI_KI Tune and plot the PI-controller integral gain.
    [result, fig] = tune_controller_gain(paper_gain_spec('pi_ki'), varargin{:});
end
