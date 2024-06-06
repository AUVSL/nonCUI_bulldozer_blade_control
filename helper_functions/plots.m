
time = out3(:,1, :);
X = out3(:,2, :);
Y = out3(:,3, :);
roll_error  = out3(:,5, :);
% depth_error = out(:,5, :);
% yaw_error   = out(:,6, :);

time = reshape(time, length(time), 1);
% X = reshape(X, length(X), 1);
% Y = reshape(Y, length(Y), 1);

% plot(time, X);

roll_error  = reshape(roll_error,  length(roll_error),  1);
% depth_error = reshape(depth_error, length(depth_error), 1);
% yaw_error   = reshape(yaw_error,   length(yaw_error),   1);

    plot(time, roll_error, 'Color', [0 0 0], 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Roll Error (m)');
    [rmse, me] = courseErrors(roll_error)

% if(pitch_bool == 1)
%     plot(time, depth_error, 'Color', [0 0 0], 'LineWidth', 2);
%     xlabel('Time (s)'); ylabel('Depth (m)');
%     [rmse, me] = courseErrors(depth_error)
% end
% if(yaw_bool == 1)
%     plot(time, yaw_error, 'Color', [0 0 0], 'LineWidth', 2);
%     xlabel('Time (s)'); ylabel('Yaw Error (rad)');
%     [rmse, me] = courseErrors(yaw_error)
% end