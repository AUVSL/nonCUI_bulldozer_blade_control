function [rmse, me] = plots(yaw_bool, pitch_bool, roll_bool, out)
time        = out(:,1, :);
x_pos       = out(:,2, :);
y_pos       = out(:,3, :);
yaw_error   = out(:,4, :);
pitch_error = out(:,5, :);
roll_error  = out(:,6, :);

time        = reshape(time,        length(time),        1);
x_pos       = reshape(x_pos,       length(x_pos),       1);
y_pos       = reshape(y_pos,       length(y_pos),       1);
yaw_error   = reshape(yaw_error,   length(yaw_error),   1);
pitch_error = reshape(pitch_error, length(pitch_error), 1);
roll_error  = reshape(roll_error,  length(roll_error),  1);
if(yaw_bool == 1)
    plot(time, yaw_error, 'Color', [0 0 0], 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Depth (m)');
    [rmse, me] = courseErrors(yaw_error)
end
if(pitch_bool == 1)
    plot(time, pitch_error, 'Color', [0 0 0], 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Pitch Error (rad)');
    [rmse, me] = courseErrors(pitch_error)
end
if(roll_bool == 1)
    plot(time, roll_error, 'Color', [0 0 0], 'LineWidth', 2);
    xlabel('Time (s)'); ylabel('Roll Error (rad)');
    [rmse, me] = courseErrors(roll_error)
end