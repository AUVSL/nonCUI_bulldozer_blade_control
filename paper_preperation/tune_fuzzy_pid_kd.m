function [result, fig] = tune_fuzzy_pid_kd(varargin)
%TUNE_FUZZY_PID_KD Plot individual fuzzy-PID derivative-gain sensitivity.
%   For the paper's tied I/D tuning coordinate, use
%   TUNE_FUZZY_PID_ID_PAIR instead.
    [result, fig] = tune_controller_gain( ...
        paper_gain_spec('fuzzy_pid_kd'), varargin{:});
end
