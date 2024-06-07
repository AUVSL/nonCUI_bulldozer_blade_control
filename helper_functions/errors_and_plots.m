function [rmse_r, me_r, rmse_d, me_d, rmse_y, me_y] = errors_and_plots(sim_out)
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

[rmse_r, me_r] = errors( roll_error);
[rmse_d, me_d] = errors(depth_error);
[rmse_y, me_y] = errors(  yaw_error);

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
