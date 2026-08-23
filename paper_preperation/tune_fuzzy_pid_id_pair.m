function [result, fig] = tune_fuzzy_pid_id_pair(varargin)
%TUNE_FUZZY_PID_ID_PAIR Tune paired fuzzy-PID I/D output gains as in the paper.
    [result, fig] = tune_controller_gain( ...
        paper_gain_spec('fuzzy_pid_id_pair'), varargin{:});
end
