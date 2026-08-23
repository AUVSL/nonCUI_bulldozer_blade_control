function [result, fig] = tune_proposed_input_scale(varargin)
%TUNE_PROPOSED_INPUT_SCALE Tune the proposed fuzzy input scaling gain.
    [result, fig] = tune_controller_gain( ...
        paper_gain_spec('proposed_input_scale'), varargin{:});
end
