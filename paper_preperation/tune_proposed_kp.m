function [result, fig] = tune_proposed_kp(varargin)
%TUNE_PROPOSED_KP Tune and plot the proposed fuzzy-controller output gain.
    [result, fig] = tune_controller_gain( ...
        paper_gain_spec('proposed_kp'), varargin{:});
end
