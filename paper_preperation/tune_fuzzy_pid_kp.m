function [result, fig] = tune_fuzzy_pid_kp(varargin)
%TUNE_FUZZY_PID_KP Tune and plot the fuzzy-PID proportional output gain.
    [result, fig] = tune_controller_gain( ...
        paper_gain_spec('fuzzy_pid_kp'), varargin{:});
end
