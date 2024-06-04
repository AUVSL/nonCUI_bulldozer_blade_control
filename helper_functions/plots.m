function [rmse, me] = plots(roll_bool, pitch_bool, yaw_bool, out)
time        = out(:,1, :);
roll_bdy    = out(:,2, :);
pitch_bdy   = out(:,3, :);
roll_error  = out(:,4, :);
depth_error = out(:,5, :);
yaw_error   = out(:,6, :);

time        = reshape(time,        length(time),        1);
roll_bdy    = reshape(roll_bdy,    length(roll_bdy),    1);
pitch_bdy   = reshape(pitch_bdy,   length(pitch_bdy),   1);
roll_error  = reshape(roll_error,  length(roll_error),  1);
depth_error = reshape(depth_error, length(depth_error), 1);
yaw_error   = reshape(yaw_error,   length(yaw_error),   1);
if(roll_bool == 1)
    plot(time, roll_error, 'Color', [0 0 0], 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Roll Error (m)');
    [rmse, me] = courseErrors(roll_error)
end
if(pitch_bool == 1)
    plot(time, depth_error, 'Color', [0 0 0], 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Depth (m)');
    [rmse, me] = courseErrors(depth_error)
end
if(yaw_bool == 1)
    plot(time, yaw_error, 'Color', [0 0 0], 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Yaw Error (rad)');
    [rmse, me] = courseErrors(yaw_error)
end