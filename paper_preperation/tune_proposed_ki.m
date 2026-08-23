function [result, fig] = tune_proposed_ki(varargin)
%TUNE_PROPOSED_KI Tune the proposed fuzzy-controller integral gain.
%   This is an experimental tuning coordinate: the paper's proposed
%   controller used KiProp = 0 and tuned only its input scale and output
%   proportional gain.
    [result, fig] = tune_controller_gain( ...
        paper_gain_spec('proposed_ki'), varargin{:});
end
