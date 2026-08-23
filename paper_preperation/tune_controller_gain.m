function [result, fig] = tune_controller_gain(spec, varargin)
%TUNE_CONTROLLER_GAIN Paper-style logarithmic and linear gain search.
%
%   [RESULT, FIG] = TUNE_CONTROLLER_GAIN(SPEC) runs noisy compact-soil
%   Case 1. It first evaluates a base-10 logarithmic gain grid, then performs
%   local linear refinements around the best point. Depth RMSE is the default
%   selection objective because it is the only tuning error named in the
%   paper; the archived tuning workbook's roll-plus-depth audit score is
%   also recorded for comparison.
%
%   The search stops when an interior selected point's depth RMSE changes
%   by no more than DepthTolerance_mm between refinements. The default
%   0.001 mm threshold is an explicit, reproducible interpretation of the
%   paper's ambiguous "third least significant figure" stopping rule.
%
%   Name-value options:
%     'Regenerate'            Run simulations instead of replotting saved data.
%     'LogMagnitudes'         Positive coarse-grid gain magnitudes.
%     'LinearPoints'          Points in each linear refinement (default 11).
%     'MaxLinearRefinements'  Maximum refinement passes (default 3).
%     'DepthTolerance_mm'     Depth-RMSE convergence tolerance (default 0.001).
%     'Objective'             'depth' (default) or 'roll_depth_sum'.
%     'UseFastRestart'        Reuse the compiled model (default true).
%     'OutputDirectory'       Override the generated output directory.

    validate_spec(spec);
    options = parse_options(spec, varargin{:});
    tuning_dir = fileparts(mfilename('fullpath'));
    project_dir = fileparts(tuning_dir);
    if ~isfile(fullfile(project_dir, 'simulation_3d.slx'))
        error(['The tuning files must remain directly inside the project''s ' ...
            'paper_preperation directory.']);
    end

    original_path = path;
    path_cleanup = onCleanup(@() path(original_path));
    addpath(genpath(fullfile(project_dir, 'controllers')), '-begin');
    addpath(genpath(fullfile(project_dir, 'sim_helper_functions')), '-begin');
    assert_project_resolution(project_dir, 'blade_ang.fis');
    assert_project_resolution(project_dir, 'blade_height.fis');
    assert_project_resolution(project_dir, ...
        'simulation_stopping_and_state_loading.m');

    output_dir = resolve_output_directory(project_dir, spec.ID, ...
        options.OutputDirectory);
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end
    data_file = fullfile(output_dir, [spec.ID, '_tuning_data.mat']);

    if options.Regenerate
        [evaluations, converged, stop_reason] = run_search( ...
            project_dir, spec, options);
        best = best_evaluation(evaluations);
        summary_table = make_summary_table(evaluations);
        result = struct( ...
            'Spec', spec, ...
            'Options', options, ...
            'Evaluations', evaluations, ...
            'SummaryTable', summary_table, ...
            'Best', best, ...
            'Converged', converged, ...
            'StopReason', stop_reason, ...
            'OutputDirectory', output_dir);
        save(data_file, 'result', '-v7.3');
    else
        if ~isfile(data_file)
            error('Saved tuning data was not found: %s', data_file);
        end
        saved = load(data_file, 'result');
        result = saved.result;
        result.OutputDirectory = output_dir;
        spec = result.Spec;
    end

    fprintf('\n%s - %s tuning results\n', spec.ControllerName, spec.GainLabel);
    disp(result.SummaryTable);
    fprintf(['Selected %s magnitude: %.9g (applied gain %.9g)\n' ...
             'RMSE: roll %.6f mrad, depth %.6f mm, yaw %.6f mrad\n' ...
             'Objective (%s): %.6f\n%s\n'], ...
            spec.GainLabel, result.Best.GainMagnitude, ...
            result.Best.AppliedGain, result.Best.RollRMSE_mrad, ...
            result.Best.DepthRMSE_mm, result.Best.YawRMSE_mrad, ...
            objective_label(result.Options.Objective), ...
            result.Best.Objective, result.StopReason);

    csv_file = fullfile(output_dir, [spec.ID, '_error_vs_gain.csv']);
    writetable(result.SummaryTable, csv_file);
    fig = plot_error_vs_gain(result);
    savefig(fig, fullfile(output_dir, [spec.ID, '_error_vs_gain.fig']));
    exportgraphics(fig, ...
        fullfile(output_dir, [spec.ID, '_error_vs_gain.png']), ...
        'Resolution', 300);
    saveas(fig, fullfile(output_dir, [spec.ID, '_error_vs_gain.svg']));

    fprintf('Saved tuning data and plots to: %s\n', output_dir);
end

function options = parse_options(spec, varargin)
    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'Regenerate', true, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'LogMagnitudes', spec.LogMagnitudes, ...
        @(x) isnumeric(x) && isvector(x) && numel(x) >= 2 && ...
        all(isfinite(x)) && all(x > 0));
    addParameter(parser, 'LinearPoints', 11, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && ...
        x >= 3 && x == floor(x));
    addParameter(parser, 'MaxLinearRefinements', 3, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && ...
        x >= 1 && x == floor(x));
    addParameter(parser, 'DepthTolerance_mm', 0.001, ...
        @(x) isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0);
    addParameter(parser, 'Objective', 'depth', ...
        @(x) any(strcmpi(string(x), ["depth", "roll_depth_sum"])));
    addParameter(parser, 'UseFastRestart', true, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'OutputDirectory', "", ...
        @(x) ischar(x) || (isstring(x) && isscalar(x)));
    parse(parser, varargin{:});
    options = parser.Results;
    options.LogMagnitudes = reshape(unique(sort( ...
        options.LogMagnitudes(:))), 1, []);
    options.Objective = char(lower(string(options.Objective)));
end

function validate_spec(spec)
    required_fields = {'ID', 'ControllerName', 'ControllerIndex', ...
        'GainVariable', 'GainLabel', 'GainSign', 'NominalMagnitude', ...
        'LogMagnitudes'};
    missing_fields = required_fields(~isfield(spec, required_fields));
    if ~isempty(missing_fields)
        error('Gain specification is missing: %s', strjoin(missing_fields, ', '));
    end
    validateattributes(spec.ControllerIndex, {'numeric'}, ...
        {'scalar', 'integer', '>=', 1, '<=', 4});
    validateattributes(spec.GainSign, {'numeric'}, ...
        {'scalar', 'real', 'finite'});
    if spec.GainSign == 0
        error('GainSign must be nonzero.');
    end
    validateattributes(spec.NominalMagnitude, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'positive'});
    validateattributes(spec.LogMagnitudes, {'numeric'}, ...
        {'vector', 'real', 'finite', 'positive'});

    allowed_gain_variables = { ...
        {'KpP'}, ...
        {'KpPI', 'KiPI'}, ...
        {'KpCong', 'KiCong', 'KdCong', 'KinputCong'}, ...
        {'KpProp', 'KiProp', 'KdProp', 'KinputProp'}};
    gain_variables = gain_variable_names(spec);
    if ~all(ismember(gain_variables, ...
            allowed_gain_variables{spec.ControllerIndex}))
        error(['Gain variable(s) %s do not belong to controller %d. ' ...
            'Allowed variables: %s.'], strjoin(gain_variables, ', '), ...
            spec.ControllerIndex, strjoin( ...
            allowed_gain_variables{spec.ControllerIndex}, ', '));
    end
end

function [evaluations, converged, stop_reason] = run_search(project_dir, spec, options)
    model_name = 'simulation_3d';
    model_file = fullfile(project_dir, [model_name, '.slx']);
    if ~isfile(model_file)
        error('Simulink model was not found: %s', model_file);
    end

    model_was_loaded = bdIsLoaded(model_name);
    if ~model_was_loaded
        load_system(model_file);
    elseif ~same_path(get_param(model_name, 'FileName'), model_file)
        error('A different %s model is already loaded: %s', ...
            model_name, get_param(model_name, 'FileName'));
    end

    original_fast_restart = get_param(model_name, 'FastRestart');
    model_cleanup         = onCleanup(@() restore_model( ...
        model_name, original_fast_restart, model_was_loaded));
    clear('simulation_stopping_and_state_loading');
    set_param(model_name, 'FastRestart', on_off(options.UseFastRestart));

    gain_variables = gain_variable_names(spec);
    fprintf('\n%s: tuning %s (%s) in paper Case 1\n', ...
        spec.ControllerName, spec.GainLabel, strjoin(gain_variables, ', '));
    fprintf('Stage 1: base-10 logarithmic grid\n');
    evaluations = evaluate_grid(model_name, project_dir, spec, ...
        options.LogMagnitudes, 'log', 0, options.UseFastRestart, ...
        options.Objective);
    stage_best = best_evaluation(evaluations);
    previous_depth_rmse = stage_best.DepthRMSE_mm;
    bounds = neighboring_bounds(options.LogMagnitudes, ...
        stage_best.GainMagnitude);

    converged = false;
    stop_reason = sprintf('Reached %d linear refinement(s).', ...
        options.MaxLinearRefinements);
    for refinement = 1:options.MaxLinearRefinements
        fprintf('Stage 2.%d: linear grid from %.9g to %.9g\n', ...
            refinement, bounds(1), bounds(2));
        linear_grid = linspace(bounds(1), bounds(2), options.LinearPoints);
        stage_evaluations = evaluate_grid(model_name, project_dir, spec, ...
            linear_grid, 'linear', refinement, options.UseFastRestart, ...
            options.Objective);
        evaluations = [evaluations; stage_evaluations]; %#ok<AGROW>
        stage_best  = best_evaluation(stage_evaluations);

        depth_change = abs(stage_best.DepthRMSE_mm - previous_depth_rmse);
        selected_at_boundary = is_grid_boundary( ...
            linear_grid, stage_best.GainMagnitude);
        fprintf('Depth-RMSE change at selected point: %.9g mm\n', depth_change);
        if selected_at_boundary
            fprintf(['Selected point is on a search boundary; ' ...
                'convergence cannot yet be declared.\n']);
            stop_reason = sprintf([ ...
                'Reached %d linear refinement(s); the selected point ' ...
                'remained on a search boundary. Expand LogMagnitudes ' ...
                'before accepting it as an optimum.'], refinement);
        elseif depth_change <= options.DepthTolerance_mm
            converged = true;
            stop_reason = sprintf([ ...
                'Converged after linear refinement %d: depth-RMSE change ' ...
                '%.9g mm <= %.9g mm.'], refinement, depth_change, ...
                options.DepthTolerance_mm);
            break
        else
            stop_reason = sprintf([ ...
                'Reached %d linear refinement(s): depth-RMSE change ' ...
                '%.9g mm > %.9g mm.'], refinement, depth_change, ...
                options.DepthTolerance_mm);
        end

        previous_depth_rmse = stage_best.DepthRMSE_mm;
        bounds = neighboring_bounds(linear_grid, stage_best.GainMagnitude);
    end
end

function evaluations = evaluate_grid(model_name, project_dir, spec, ...
        magnitudes, stage, refinement, use_fast_restart, objective_name)
    empty_evaluation = struct( ...
        'Stage', '', ...
        'Refinement', [], ...
        'GainMagnitude', [], ...
        'AppliedGain', [], ...
        'Success', false, ...
        'Message', '', ...
        'Samples', [], ...
        'FinalTime_s', [], ...
        'RollRMSE_mrad', [], ...
        'DepthRMSE_mm', [], ...
        'YawRMSE_mrad', [], ...
        'RollPlusDepth', [], ...
        'Objective', [], ...
        'Data', []);
    evaluations = repmat(empty_evaluation, numel(magnitudes), 1);

    for index = 1:numel(magnitudes)
        magnitude = magnitudes(index);
        applied_gain = sign(spec.GainSign) * magnitude;
        fprintf('  %s = %.9g (applied %.9g): ', ...
            spec.GainLabel, magnitude, applied_gain);

        evaluation = empty_evaluation;
        evaluation.Stage = stage;
        evaluation.Refinement = refinement;
        evaluation.GainMagnitude = magnitude;
        evaluation.AppliedGain = applied_gain;

        try
            if ~use_fast_restart
                clear('simulation_stopping_and_state_loading');
            end
            sim_input = make_simulation_input(model_name, project_dir, ...
                spec, applied_gain);
            sim_output = sim(sim_input);
            logged_output = sim_output.get('output');
            data = normalize_logged_data(logged_output.Data);
            if size(data, 2) ~= 10 || any(~isfinite(data), 'all')
                error('Invalid logged output size or nonfinite data.');
            end

            roll_error = data(:, 8);
            depth_error = data(:, 9);
            yaw_error = data(:, 10);
            evaluation.Success = true;
            evaluation.Samples = size(data, 1);
            evaluation.FinalTime_s = data(end, 1);
            evaluation.RollRMSE_mrad = 1000 * sqrt(mean(roll_error.^2));
            evaluation.DepthRMSE_mm = 1000 * sqrt(mean(depth_error.^2));
            evaluation.YawRMSE_mrad = 1000 * sqrt(mean(yaw_error.^2));
            evaluation.RollPlusDepth = evaluation.RollRMSE_mrad + ...
                evaluation.DepthRMSE_mm;
            evaluation.Objective = objective_value(evaluation, objective_name);
            evaluation.Data = data;
            fprintf('objective %.6f (roll %.3f, depth %.3f, yaw %.3f)\n', ...
                evaluation.Objective, evaluation.RollRMSE_mrad, ...
                evaluation.DepthRMSE_mm, evaluation.YawRMSE_mrad);
        catch exception
            evaluation.Message = exception.message;
            evaluation.Samples = 0;
            evaluation.FinalTime_s = NaN;
            evaluation.RollRMSE_mrad = NaN;
            evaluation.DepthRMSE_mm = NaN;
            evaluation.YawRMSE_mrad = NaN;
            evaluation.RollPlusDepth = NaN;
            evaluation.Objective = Inf;
            fprintf('FAILED: %s\n', exception.message);
            if use_fast_restart && bdIsLoaded(model_name)
                try
                    set_param(model_name, 'FastRestart', 'off');
                    clear('simulation_stopping_and_state_loading');
                    set_param(model_name, 'FastRestart', 'on');
                catch
                end
            end
        end
        evaluations(index) = evaluation;
    end
end

function sim_input = make_simulation_input(model_name, project_dir, spec, applied_gain)
    % Paper tuning uses compact soil with the noisy Case 1 observer.
    soil = 0.1;
    noise_power = 2e-7;
    desired_angle_rad = -0.005;
    surface_angle_rad = 0.005;
    controllerIndex1234 = spec.ControllerIndex;

    run(fullfile(project_dir, 'parameters.m'));

    padding = 0;
    desired_abg = [desired_angle_rad, 1, desired_angle_rad];
    surface_abg = [surface_angle_rad, 0, surface_angle_rad];
    bld_ang = zeros(3, 1);
    F_track = [60000; 60000];
    q = [0; 0; 0; surface_abg(1); surface_abg(2); surface_abg(3)];
    q_dot = zeros(6, 1);
    x_ICR = 0;
    v = zeros(2, 1);

    simin = [padding, F_track(1), F_track(2), q_dot(1), q_dot(2), ...
        q_dot(3), q_dot(4), q_dot(5), q_dot(6), q(1), q(2), q(3), ...
        q(4), q(5), q(6), x_ICR, v(1), v(2), bld_ang(1), ...
        bld_ang(2), bld_ang(3)];
    initial_states = [F_track(1), F_track(2), q_dot(1), q_dot(2), ...
        q_dot(3), q_dot(4), q_dot(5), q_dot(6), q(1), q(2), q(3), ...
        q(4), q(5), q(6), x_ICR, v(1), v(2), bld_ang(1), ...
        bld_ang(2), bld_ang(3)]';
    desired_angles = [padding, desired_abg];
    bt_params = [padding, B1, H, L, b, l, r, m, grav, velocity_limit, ...
        fill_distance, mu_t, mu_l, mu_ss, kb, gamma_g, beta0, surface_abg];
    vd_params = [padding, m, h, b, l, r, grav];
    v_limit = [velocity_limit, turn_vel_limit];
    initial_v = v;
    initial_q = q;

    sim_variables = struct( ...
        'controllerIndex1234', controllerIndex1234, ...
        'soil', soil, ...
        'noise_power', noise_power, ...
        'use_direct_depth', use_direct_depth, ...
        'desired_depth_override_m', desired_depth_override_m, ...
        'dt', dt, ...
        'stop_time', stop_time, ...
        'stop_distance', stop_distance, ...
        'gain', gain, ...
        'derivative_filter_samples', derivative_filter_samples, ...
        'KpP', KpP, ...
        'KpPI', KpPI, ...
        'KiPI', KiPI, ...
        'KpCong', KpCong, ...
        'KiCong', KiCong, ...
        'KdCong', KdCong, ...
        'KinputCong', KinputCong, ...
        'KpProp', KpProp, ...
        'KiProp', KiProp, ...
        'KdProp', KdProp, ...
        'KinputProp', KinputProp, ...
        'L', L, ...
        'l', l, ...
        'simin', simin, ...
        'initial_states', initial_states, ...
        'desired_angles', desired_angles, ...
        'bt_params', bt_params, ...
        'vd_params', vd_params, ...
        'v_limit', v_limit, ...
        'initial_v', initial_v, ...
        'initial_q', initial_q);

    gain_variables = gain_variable_names(spec);
    for gain_index = 1:numel(gain_variables)
        gain_variable = gain_variables{gain_index};
        if ~isfield(sim_variables, gain_variable)
            error('Unknown tunable gain variable: %s', gain_variable);
        end
        sim_variables.(gain_variable) = applied_gain;
    end

    sim_input = Simulink.SimulationInput(model_name);
    variable_names = fieldnames(sim_variables);
    for variable_index = 1:numel(variable_names)
        variable_name = variable_names{variable_index};
        sim_input = sim_input.setVariable(variable_name, ...
            sim_variables.(variable_name));
    end
end

function data = normalize_logged_data(raw_data)
    signal_count = size(raw_data, 2);
    sample_count = numel(raw_data(:, 1, :));
    data = zeros(sample_count, signal_count, 'like', raw_data);
    for signal_index = 1:signal_count
        data(:, signal_index) = reshape( ...
            raw_data(:, signal_index, :), sample_count, 1);
    end
end

function best = best_evaluation(evaluations)
    successful = [evaluations.Success];
    if ~any(successful)
        messages = string({evaluations.Message});
        error('Every gain evaluation failed:\n%s', strjoin(messages, '\n'));
    end
    candidates = evaluations(successful);
    [~, best_index] = min([candidates.Objective]);
    best = candidates(best_index);
end

function bounds = neighboring_bounds(grid, selected_value)
    grid = reshape(unique(sort(grid(:))), 1, []);
    [~, selected_index] = min(abs(grid - selected_value));
    lower_index = max(1, selected_index - 1);
    upper_index = min(numel(grid), selected_index + 1);
    if lower_index == upper_index
        if selected_index == 1
            upper_index = 2;
        else
            lower_index = numel(grid) - 1;
        end
    end
    bounds = [grid(lower_index), grid(upper_index)];
end

function tf = is_grid_boundary(grid, selected_value)
    grid = reshape(unique(sort(grid(:))), 1, []);
    [~, selected_index] = min(abs(grid - selected_value));
    tf = selected_index == 1 || selected_index == numel(grid);
end

function summary_table = make_summary_table(evaluations)
    summary_table = table( ...
        string({evaluations.Stage})', ...
        [evaluations.Refinement]', ...
        [evaluations.GainMagnitude]', ...
        [evaluations.AppliedGain]', ...
        [evaluations.Success]', ...
        [evaluations.Samples]', ...
        [evaluations.FinalTime_s]', ...
        [evaluations.RollRMSE_mrad]', ...
        [evaluations.DepthRMSE_mm]', ...
        [evaluations.YawRMSE_mrad]', ...
        [evaluations.RollPlusDepth]', ...
        [evaluations.Objective]', ...
        string({evaluations.Message})', ...
        'VariableNames', {'Stage', 'Refinement', 'GainMagnitude', ...
        'AppliedGain', 'Success', 'Samples', 'FinalTime_s', ...
        'RollRMSE_mrad', 'DepthRMSE_mm', 'YawRMSE_mrad', ...
        'RollPlusDepth', 'Objective', 'Message'});
end

function fig = plot_error_vs_gain(result)
    successful = result.SummaryTable.Success;
    table_data = result.SummaryTable(successful, :);
    log_data = table_data(table_data.Stage == "log", :);
    refinements = table_data.Refinement(table_data.Stage == "linear");
    if isempty(refinements)
        linear_data = log_data;
        linear_title = 'Linear refinement unavailable';
    else
        final_refinement = max(refinements);
        linear_data = table_data(table_data.Stage == "linear" & ...
            table_data.Refinement == final_refinement, :);
        linear_title = sprintf('Linear refinement %d', final_refinement);
    end

    log_data = sortrows(log_data, 'GainMagnitude');
    linear_data = sortrows(linear_data, 'GainMagnitude');
    fig = figure('Units', 'inches', 'Position', [1, 1, 10, 4.8], ...
        'Color', 'w', 'Name', [result.Spec.ID, ' error versus gain']);
    layout = tiledlayout(fig, 1, 2, ...
        'Padding', 'compact', 'TileSpacing', 'compact');

    first_axes = nexttile(layout);
    plot_metric_curves(first_axes, log_data, true);
    title(first_axes, 'Base-10 logarithmic grid', 'FontWeight', 'normal');

    second_axes = nexttile(layout);
    plot_metric_curves(second_axes, linear_data, false);
    title(second_axes, linear_title, 'FontWeight', 'normal');

    axes_handles = [first_axes, second_axes];
    for axes_handle = axes_handles
        add_selected_gain_line(axes_handle, result.Best.GainMagnitude);
        xlabel(axes_handle, sprintf('%s magnitude', result.Spec.GainLabel));
        ylabel(axes_handle, 'RMSE');
        grid(axes_handle, 'on');
        set(axes_handle, 'FontSize', 11, 'Box', 'off');
    end

    title(layout, sprintf('%s - %s error versus gain', ...
        result.Spec.ControllerName, result.Spec.GainLabel), ...
        'FontWeight', 'normal');
end

function plot_metric_curves(axes_handle, table_data, logarithmic_x)
    hold(axes_handle, 'on');
    x = table_data.GainMagnitude;
    if logarithmic_x
        plot_function = @semilogx;
    else
        plot_function = @plot;
    end
    plot_function(axes_handle, x, table_data.RollRMSE_mrad, ...
        '-o', 'LineWidth', 1.5, 'DisplayName', 'Roll RMSE (mrad)');
    plot_function(axes_handle, x, table_data.DepthRMSE_mm, ...
        '-s', 'LineWidth', 1.5, 'DisplayName', 'Depth RMSE (mm)');
    plot_function(axes_handle, x, table_data.YawRMSE_mrad, ...
        '--^', 'LineWidth', 1.5, 'DisplayName', 'Yaw RMSE (mrad)');
    legend(axes_handle, 'Location', 'best', 'FontSize', 9);
end

function add_selected_gain_line(axes_handle, selected_gain)
    limits = xlim(axes_handle);
    if selected_gain >= limits(1) && selected_gain <= limits(2)
        xline(axes_handle, selected_gain, '--', 'Selected', ...
            'Color', [0.75, 0.1, 0.1], 'HandleVisibility', 'off');
    end
end

function output_dir = resolve_output_directory(project_dir, id, requested_dir)
    if strlength(requested_dir) == 0
        output_dir = fullfile(project_dir, 'paper_preperation', ...
            'generated_tuning', id);
    else
        output_dir = char(requested_dir);
        if ~is_absolute_path(output_dir)
            output_dir = fullfile(project_dir, output_dir);
        end
    end
end

function restore_model(model_name, original_fast_restart, model_was_loaded)
    if ~bdIsLoaded(model_name)
        return
    end
    try
        set_param(model_name, 'FastRestart', original_fast_restart);
    catch
    end
    if ~model_was_loaded
        close_system(model_name, 0);
    end
end

function value = on_off(tf)
    if tf
        value = 'on';
    else
        value = 'off';
    end
end

function value = objective_value(evaluation, objective_name)
    switch lower(objective_name)
        case 'depth'
            value = evaluation.DepthRMSE_mm;
        case 'roll_depth_sum'
            value = evaluation.RollPlusDepth;
        otherwise
            error('Unknown tuning objective: %s', objective_name);
    end
end

function label = objective_label(objective_name)
    switch lower(objective_name)
        case 'depth'
            label = 'depth RMSE (mm)';
        case 'roll_depth_sum'
            label = 'roll RMSE + depth RMSE';
        otherwise
            label = objective_name;
    end
end

function gain_variables = gain_variable_names(spec)
    gain_variables = {spec.GainVariable};
    if isfield(spec, 'AdditionalGainVariables') && ...
            ~isempty(spec.AdditionalGainVariables)
        gain_variables = [gain_variables, spec.AdditionalGainVariables];
    end
end

function assert_project_resolution(project_dir, file_name)
    resolved_file = which(file_name);
    if isempty(resolved_file)
        error('Required project file is not on the MATLAB path: %s', file_name);
    end
    project_prefix = [lower(canonical_path(project_dir)), filesep];
    if ~startsWith(lower(canonical_path(resolved_file)), project_prefix)
        error('%s resolves outside this project: %s', file_name, resolved_file);
    end
end

function tf = is_absolute_path(file_path)
    tf = ~isempty(regexp(file_path, '^[A-Za-z]:[\\/]|^[/\\]{2}|^/', 'once'));
end

function tf = same_path(first_path, second_path)
    tf = strcmpi(canonical_path(first_path), canonical_path(second_path));
end

function result = canonical_path(file_path)
    result = char(java.io.File(file_path).getCanonicalPath());
end
