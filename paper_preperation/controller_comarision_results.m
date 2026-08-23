function [runs, rmse_table, fig] = controller_comarision_results(varargin)
%GENERATE_FIGURE11_RESULTS Run and plot the experiments shown in Figure 11.
%
%   generate_figure11_results runs all three paper cases with the P, PI,
%   fuzzy PID, and proposed fuzzy controllers. It saves the raw logged data,
%   an RMSE table, and a 3-by-3 comparison figure.
%
%   generate_figure11_results('Regenerate', false) reloads previously saved
%   data and only rebuilds the table and figure.
%
%   Name-value options:
%     'Regenerate'      - Run Simulink before plotting (default: true).
%     'OutputDirectory' - Destination for generated files. The default is
%                         paper_preperation/generated_figure11.

    options = parse_options(varargin{:});

    script_dir = fileparts(mfilename('fullpath'));
    project_dir = fileparts(script_dir);
    original_path = path;
    path_cleanup = onCleanup(@() path(original_path)); %#ok<NASGU>
    addpath(genpath(fullfile(project_dir, 'controllers')), '-begin');
    addpath(genpath(fullfile(project_dir, 'sim_helper_functions')), '-begin');
    assert_project_resolution(project_dir, 'blade_ang.fis');
    assert_project_resolution(project_dir, 'blade_height.fis');
    assert_project_resolution(project_dir, ...
        'simulation_stopping_and_state_loading.m');

    if strlength(options.OutputDirectory) == 0
        output_dir = fullfile(script_dir, 'generated_figure11');
    else
        output_dir = char(options.OutputDirectory);
        if ~is_absolute_path(output_dir)
            output_dir = fullfile(project_dir, output_dir);
        end
    end
    if ~isfolder(output_dir)
        mkdir(output_dir);
    end

    data_file = fullfile(output_dir, 'figure11_run_data.mat');
    case_defs = paper_cases();
    controller_defs = paper_controllers();

    if options.Regenerate
        runs = run_experiments(project_dir, case_defs, controller_defs);
        rmse_table = make_rmse_table(runs);
        save(data_file, 'runs', 'rmse_table', 'case_defs', ...
            'controller_defs', '-v7.3');
    else
        if ~isfile(data_file)
            error('Saved Figure 11 data was not found: %s', data_file);
        end
        saved = load(data_file, 'runs', 'case_defs', 'controller_defs');
        runs = saved.runs;
        case_defs = saved.case_defs;
        controller_defs = saved.controller_defs;
        rmse_table = make_rmse_table(runs);
    end

    fprintf('\nRMSE summary (true/non-observer errors)\n');
    disp(rmse_table);
    writetable(rmse_table, fullfile(output_dir, 'figure11_rmse.csv'));

    fig = plot_figure11(runs, case_defs, controller_defs);
    savefig(fig, fullfile(output_dir, 'figure11_results.fig'));
    exportgraphics(fig, fullfile(output_dir, 'figure11_results.png'), ...
        'Resolution', 300);
    saveas(fig, fullfile(output_dir, 'figure11_results.svg'));

    fprintf('Raw data: %s\n', data_file);
    fprintf('RMSE table: %s\n', ...
        fullfile(output_dir, 'figure11_rmse.csv'));
    fprintf('Figure files: %s\n', output_dir);
end

function options = parse_options(varargin)
    parser = inputParser;
    parser.FunctionName = mfilename;
    addParameter(parser, 'Regenerate', true, ...
        @(x) islogical(x) && isscalar(x));
    addParameter(parser, 'OutputDirectory', "", ...
        @(x) ischar(x) || (isstring(x) && isscalar(x)));
    parse(parser, varargin{:});
    options = parser.Results;
end

function case_defs = paper_cases()
    case_defs(1) = struct( ...
        'Name', 'Case 1 - compact soil, noisy observer', ...
        'ShortName', 'Case 1: compact + noise', ...
        'Soil', 0.1, ...
        'NoisePower', 2e-7, ...
        'DesiredDepth', -0.03, ...
        'DesiredAngle', -0.005, ...
        'SurfaceAngle', 0.005);
    case_defs(2) = struct( ...
        'Name', 'Case 2 - loose soil, noisy observer', ...
        'ShortName', 'Case 2: loose + noise', ...
        'Soil', 0.9, ...
        'NoisePower', 2e-7, ...
        'DesiredDepth', -0.04, ...
        'DesiredAngle', -0.003, ...
        'SurfaceAngle', 0.003);
    case_defs(3) = struct( ...
        'Name', 'Case 3 - compact soil, no observer noise', ...
        'ShortName', 'Case 3: compact + no noise', ...
        'Soil', 0.1, ...
        'NoisePower', 0, ...
        'DesiredDepth', -0.03, ...
        'DesiredAngle', -0.005, ...
        'SurfaceAngle', 0.005);
end

function controller_defs = paper_controllers()
    controller_defs(1) = struct('Name', 'P', 'Index', 1, ...
        'LineStyle', '-.', 'Color', [0, 0, 0]);
    controller_defs(2) = struct('Name', 'PI', 'Index', 2, ...
        'LineStyle', ':', 'Color', [0, 0, 0]);
    controller_defs(3) = struct('Name', 'Fzy PID', 'Index', 3, ...
        'LineStyle', '--', 'Color', [0, 0, 0]);
    controller_defs(4) = struct('Name', 'Fzy (Prop.)', 'Index', 4, ...
        'LineStyle', '-', 'Color', [0.4, 0.4, 0.4]);
end

function runs = run_experiments(project_dir, case_defs, controller_defs)
    model_name = 'simulation_3d';
    model_file = fullfile(project_dir, [model_name, '.slx']);
    if ~isfile(model_file)
        error('Simulink model was not found: %s', model_file);
    end

    model_was_loaded = bdIsLoaded(model_name);
    if ~model_was_loaded
        load_system(model_file);
    else
        loaded_file = get_param(model_name, 'FileName');
        if ~same_path(loaded_file, model_file)
            error('A different %s model is already loaded: %s', ...
                model_name, loaded_file);
        end
    end
    original_fast_restart = get_param(model_name, 'FastRestart');
    model_cleanup = onCleanup(@() restore_model( ...
        model_name, original_fast_restart, model_was_loaded)); %#ok<NASGU>
    set_param(model_name, 'FastRestart', 'off');

    empty_run = struct( ...
        'CaseIndex', [], ...
        'CaseName', '', ...
        'ControllerIndex', [], ...
        'ControllerName', '', ...
        'PaperOutputNumber', [], ...
        'Data', [], ...
        'Time', [], ...
        'DepthError', [], ...
        'RollError', [], ...
        'YawError', [], ...
        'RollRMSE_rad', [], ...
        'DepthRMSE_m', [], ...
        'YawRMSE_rad', []);
    runs = repmat(empty_run, numel(case_defs), numel(controller_defs));

    total_runs = numel(runs);
    run_number = 0;
    for case_index = 1:numel(case_defs)
        for controller_number = 1:numel(controller_defs)
            run_number = run_number + 1;
            case_def = case_defs(case_index);
            controller_def = controller_defs(controller_number);
            fprintf('[%02d/%02d] %s - %s\n', run_number, total_runs, ...
                case_def.Name, controller_def.Name);

            sim_input = make_simulation_input(model_name, project_dir, ...
                case_def, controller_def.Index);
            clear('simulation_stopping_and_state_loading');
            sim_output = sim(sim_input);
            logged_output = sim_output.get('output');
            data = normalize_logged_data(logged_output.Data);
            if size(data, 2) ~= 10 || any(~isfinite(data), 'all')
                error('Run %d returned invalid output with size %s.', ...
                    run_number, mat2str(size(data)));
            end
            [time, depth_error, roll_error, yaw_error] = unpack_errors(data);

            roll_rmse = sqrt(mean(roll_error.^2));
            depth_rmse = sqrt(mean(depth_error.^2));
            yaw_rmse = sqrt(mean(yaw_error.^2));

            runs(case_index, controller_number) = struct( ...
                'CaseIndex', case_index, ...
                'CaseName', case_def.Name, ...
                'ControllerIndex', controller_def.Index, ...
                'ControllerName', controller_def.Name, ...
                'PaperOutputNumber', paper_output_number(case_index, ...
                    controller_def.Index), ...
                'Data', data, ...
                'Time', time, ...
                'DepthError', depth_error, ...
                'RollError', roll_error, ...
                'YawError', yaw_error, ...
                'RollRMSE_rad', roll_rmse, ...
                'DepthRMSE_m', depth_rmse, ...
                'YawRMSE_rad', yaw_rmse);

            fprintf(['         RMSE: roll = %.3f mrad, depth = %.3f mm, ' ...
                     'yaw = %.3f mrad (%d samples, %.3f s)\n'], ...
                    1000 * roll_rmse, 1000 * depth_rmse, ...
                    1000 * yaw_rmse, numel(time), time(end));
        end
    end
end

function sim_input = make_simulation_input(model_name, project_dir, case_def, controller_index)
    soil = case_def.Soil;
    noise_power = case_def.NoisePower;
    desired_depth_m = case_def.DesiredDepth;
    desired_angle_rad = case_def.DesiredAngle;
    surface_angle_rad = case_def.SurfaceAngle;
    controllerIndex1234 = controller_index;

    run(fullfile(project_dir, 'parameters.m'));

    padding = 0;
    desired_depth = [padding, desired_depth_m];
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
        'KpProp', KpProp, ...
        'KiProp', KiProp, ...
        'KdProp', KdProp, ...
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

    sim_input = Simulink.SimulationInput(model_name);
    variable_names = fieldnames(sim_variables);
    for variable_index = 1:numel(variable_names)
        variable_name = variable_names{variable_index};
        sim_input = sim_input.setVariable(variable_name, ...
            sim_variables.(variable_name));
    end
end

function [time, depth_error, roll_error, yaw_error] = unpack_errors(data)
    if size(data, 2) < 10
        error('Expected at least 10 logged columns; received %d.', size(data, 2));
    end
    sample_count = size(data, 1);
    time = reshape(data(:, 1, :), sample_count, 1);
    roll_error = reshape(data(:, 8, :), sample_count, 1);
    depth_error = reshape(data(:, 9, :), sample_count, 1);
    yaw_error = reshape(data(:, 10, :), sample_count, 1);
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

function rmse_table = make_rmse_table(runs)
    row_count = numel(runs);
    case_number = zeros(row_count, 1);
    case_name = strings(row_count, 1);
    controller = strings(row_count, 1);
    paper_output = zeros(row_count, 1);
    sample_count = zeros(row_count, 1);
    final_time_s = zeros(row_count, 1);
    roll_rmse_mrad = zeros(row_count, 1);
    depth_rmse_mm = zeros(row_count, 1);
    yaw_rmse_mrad = zeros(row_count, 1);

    row = 0;
    for case_index = 1:size(runs, 1)
        for controller_index = 1:size(runs, 2)
            row = row + 1;
            run_result = runs(case_index, controller_index);
            case_number(row) = run_result.CaseIndex;
            case_name(row) = string(run_result.CaseName);
            controller(row) = string(run_result.ControllerName);
            paper_output(row) = run_result.PaperOutputNumber;
            sample_count(row) = numel(run_result.Time);
            final_time_s(row) = run_result.Time(end);
            roll_rmse_mrad(row) = 1000 * run_result.RollRMSE_rad;
            depth_rmse_mm(row) = 1000 * run_result.DepthRMSE_m;
            yaw_rmse_mrad(row) = 1000 * run_result.YawRMSE_rad;
        end
    end

    rmse_table = table(case_number, case_name, controller, paper_output, ...
        sample_count, final_time_s, roll_rmse_mrad, depth_rmse_mm, ...
        yaw_rmse_mrad, ...
        'VariableNames', {'Case', 'CaseName', 'Controller', ...
        'PaperOutput', 'Samples', 'FinalTime_s', 'RollRMSE_mrad', ...
        'DepthRMSE_mm', 'YawRMSE_mrad'});
end

function fig = plot_figure11(runs, case_defs, controller_defs)
    fig = figure('Units', 'inches', 'Position', [1, 1, 10, 5], ...
        'Color', 'w', 'Name', 'Figure 11 controller comparison');
    layout = tiledlayout(fig, 3, 3, ...
        'Padding', 'compact', 'TileSpacing', 'compact');

    signal_fields = {'DepthError', 'RollError', 'YawError'};
    y_labels = {'Dp Er (m)', 'Rl Er (rad)', 'Yw Er (rad)'};
    legend_handles = gobjects(1, numel(controller_defs));

    for row = 1:3
        for case_index = 1:numel(case_defs)
            tile_index = (row - 1) * numel(case_defs) + case_index;
            axes_handle = nexttile(layout, tile_index);
            hold(axes_handle, 'on');

            for controller_index = 1:numel(controller_defs)
                run_result = runs(case_index, controller_index);
                line_handle = plot(axes_handle, run_result.Time, ...
                    run_result.(signal_fields{row}), ...
                    'LineStyle', controller_defs(controller_index).LineStyle, ...
                    'Color', controller_defs(controller_index).Color, ...
                    'LineWidth', 2);
                if row == 1 && case_index == 1
                    legend_handles(controller_index) = line_handle;
                end
            end

            if case_index == 1
                ylabel(axes_handle, y_labels{row});
            else
                axes_handle.YTickLabel = [];
            end
            set(axes_handle, 'FontSize', 12, 'Box', 'off');
        end
    end

    legend_labels = {controller_defs.Name};
    legend_handle = legend(legend_handles, legend_labels, ...
        'Orientation', 'horizontal', 'NumColumns', numel(controller_defs), ...
        'FontSize', 14);
    legend_handle.Layout.Tile = 'north';
    xlabel(layout, 'Time (s)', 'FontSize', 12);
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

function output_number = paper_output_number(case_index, controller_index)
    paper_output_offset = [0, 8, 4];
    output_number = paper_output_offset(case_index) + controller_index;
end

function assert_project_resolution(project_dir, file_name)
    resolved_file = which(file_name);
    if isempty(resolved_file)
        error('Required project file is not on the MATLAB path: %s', file_name);
    end
    if ~startsWith(lower(canonical_path(resolved_file)), ...
            [lower(canonical_path(project_dir)), filesep])
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
