function [result, fig] = tune_fuzzy_pid_input_scale(varargin)
%TUNE_FUZZY_PID_INPUT_SCALE Tune the shared fuzzy-PID input scaling gain.
    [result, fig] = tune_controller_gain( ...
        paper_gain_spec('fuzzy_pid_input_scale'), varargin{:});
end
