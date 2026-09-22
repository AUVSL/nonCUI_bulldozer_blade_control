function [rmse_r, me_r, rmse_d, me_d, rmse_y, me_y, ss_depth_bound] = errors_and_plots(sim_out, ss_window_fraction)
    % Treat the final 20% of samples as steady state unless specified.
    if nargin < 2
        ss_window_fraction = 0.70;
    end
    validateattributes(ss_window_fraction, {'numeric'}, ...
        {'scalar', 'real', 'finite', '>', 0, '<=', 1}, ...
        mfilename, 'ss_window_fraction');

    % unpack the data from the simulation
    time        = reshape(sim_out(:, 1, :),  length(sim_out(:, 1, :)),  1);
    body_x      = reshape(sim_out(:, 2, :),  length(sim_out(:, 2, :)),  1);
    body_y      = reshape(sim_out(:, 3, :),  length(sim_out(:, 3, :)),  1);
    % body_z      = reshape(sim_out(:, 4, :),  length(sim_out(:, 4, :)),  1);
    body_roll   = reshape(sim_out(:, 5, :),  length(sim_out(:, 5, :)),  1);
    % body_pitch  = reshape(sim_out(:, 6, :),  length(sim_out(:, 6, :)),  1);
    body_yaw    = reshape(sim_out(:, 7, :),  length(sim_out(:, 7, :)),  1);
    roll_error  = reshape(sim_out(:, 8, :),  length(sim_out(:, 8, :)),  1);
    depth_error = reshape(sim_out(:, 9, :),  length(sim_out(:, 9, :)),  1);
    yaw_error   = reshape(sim_out(:,10, :),  length(sim_out(:,10, :)),  1);
    
    % calulate the Root Mean Squared Errors (rmse) and Maximum Errors (me)
    [rmse_r, me_r] = errors( roll_error);
    [rmse_d, me_d] = errors(depth_error);
    [rmse_y, me_y] = errors(  yaw_error);

    % Empirical steady-state depth-error bound over the requested tail window.
    sample_count = numel(depth_error);
    ss_start_idx = max(1, floor((1 - ss_window_fraction) * sample_count) + 1);
    ss_depth_bound = max(abs(depth_error(ss_start_idx:end)));
    fprintf(['Depth steady-state error bound over the final %.0f%% ' ...
             '(t >= %.3f s): |e_depth| <= %.6g m (%.3f mm)\n'], ...
            100 * ss_window_fraction, time(ss_start_idx), ...
            ss_depth_bound, 1000 * ss_depth_bound);
    
    % plot run information in a 3 by 2 tiles array
    tiledlayout(3,2)
    nexttile;
    plot(body_x, body_y,  '-', 'Color', [0 0 0], 'LineWidth', 2);
    xlabel('Body X (m)'); ylabel('Body Y (m)');
    
    nexttile;
    plot(time, body_roll,  '-', 'Color', [0 0 0], 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Body Roll (rad)');
    
    nexttile;
    plot(time, body_yaw,  '-', 'Color', [0 0 0], 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Body Yaw (rad)');
    
    nexttile;
    plot(time, roll_error,  '-', 'Color', [0 0 0], 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Roll Error (rad)');
    
    nexttile;
    plot(time, depth_error,  '-', 'Color', [0 0 0], 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Depth Error (m)');
    
    nexttile;
    plot(time, yaw_error,  '-', 'Color', [0 0 0], 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Yaw Error (rad)');
end 

% a helper function for calculating the errors
function [rmse, me] = errors(x)
    [n,~] = size(x);
    rmse = sqrt(sum(x.^2)/n);
    me = max(abs(x));
end
