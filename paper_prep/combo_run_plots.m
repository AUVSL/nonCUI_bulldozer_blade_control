clc
time1        = reshape(out1(:, 1,:), length(out1(:, 1,:)), 1);
roll_bdy1    = reshape(out1(:, 5,:), length(out1(:, 5,:)), 1);
pitch_bdy1   = reshape(out1(:, 7,:), length(out1(:, 7,:)), 1);
roll_error1  = reshape(out1(:, 8,:), length(out1(:, 8,:)), 1);
depth_error1 = reshape(out1(:, 9,:), length(out1(:, 9,:)), 1);
yaw_error1   = reshape(out1(:,10,:), length(out1(:,10,:)), 1);

time2        = reshape(out2(:, 1,:), length(out2(:, 1,:)), 1);
roll_bdy2    = reshape(out2(:, 5,:), length(out2(:, 5,:)), 1);
pitch_bdy2   = reshape(out2(:, 7,:), length(out2(:, 7,:)), 1);
roll_error2  = reshape(out2(:, 8,:), length(out2(:, 8,:)), 1);
depth_error2 = reshape(out2(:, 9,:), length(out2(:, 9,:)), 1);
yaw_error2   = reshape(out2(:,10,:), length(out2(:,10,:)), 1);

time3        = reshape(out3(:, 1,:), length(out3(:, 1,:)), 1);
roll_bdy3    = reshape(out3(:, 5,:), length(out3(:, 5,:)), 1);
pitch_bdy3   = reshape(out3(:, 7,:), length(out3(:, 7,:)), 1);
roll_error3  = reshape(out3(:, 8,:), length(out3(:, 8,:)), 1);
depth_error3 = reshape(out3(:, 9,:), length(out3(:, 9,:)), 1);
yaw_error3   = reshape(out3(:,10,:), length(out3(:,10,:)), 1);

time4        = reshape(out4(:, 1,:), length(out4(:, 1,:)), 1);
roll_bdy4    = reshape(out4(:, 5,:), length(out4(:, 5,:)), 1);
pitch_bdy4   = reshape(out4(:, 7,:), length(out4(:, 7,:)), 1);
roll_error4  = reshape(out4(:, 8,:), length(out4(:, 8,:)), 1);
depth_error4 = reshape(out4(:, 9,:), length(out4(:, 9,:)), 1);
yaw_error4   = reshape(out4(:,10,:), length(out4(:,10,:)), 1);

time5        = reshape(out5(:, 1,:), length(out5(:, 1,:)), 1);
roll_bdy5    = reshape(out5(:, 5,:), length(out5(:, 5,:)), 1);
pitch_bdy5   = reshape(out5(:, 7,:), length(out5(:, 7,:)), 1);
roll_error5  = reshape(out5(:, 8,:), length(out5(:, 8,:)), 1);
depth_error5 = reshape(out5(:, 9,:), length(out5(:, 9,:)), 1);
yaw_error5   = reshape(out5(:,10,:), length(out5(:,10,:)), 1);


tiledlayout(5,1)
ax1 = nexttile;
plot(ax1, time1, roll_error1,  '-.', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax1, time2, roll_error2,   '--', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax1, time3, roll_error3,   '-', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax1, time4, roll_error5,   ':', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax1, time4, roll_error5,   '-', 'Color', [0.1 0.1 0.1], 'LineWidth', 2);
set(gca,"FontSize", 11)
xlabel('Time (s)', 'FontSize', 12); ylabel('Roll Error (rad)', 'FontSize',12);

hold off

legend({'depth1','depth2', 'roll', 'yaw', 'combo'}, 'Location','northoutside', ...
    'Orientation', 'horizontal', 'NumColumns', 3, FontSize=11)

ax2 = nexttile;
plot(ax2, time1, depth_error1,  '-.', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax2, time2, depth_error2,   '--', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax2, time3, depth_error3,   '-', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax2, time4, depth_error5,   ':', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax2, time4, depth_error5,   '-', 'Color', [0.1 0.1 0.1], 'LineWidth', 2);
set(gca,"FontSize", 11)
xlabel('Time (s)', 'FontSize', 12); ylabel('Depth (m)', 'FontSize', 12);
axis([0 0.4 -0.32 0 ])
hold off

ax3 = nexttile;
plot(ax3, time1, yaw_error1,  '-.', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax3, time2, yaw_error2,   '--', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax3, time3, yaw_error3,   '-', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax3, time4, yaw_error5,   ':', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax3, time4, yaw_error5,   '-', 'Color', [0.1 0.1 0.1], 'LineWidth', 2);
set(gca,"FontSize", 11)
xlabel('Time (s)', 'FontSize', 12); ylabel('Yaw Error (rad)', 'FontSize', 12);
hold off

ax4 = nexttile;
plot(ax4, time1, roll_bdy1,  '-.', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax4, time2, roll_bdy2,   '--', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax4, time3, roll_bdy3,   '-', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax4, time4, roll_bdy5,   ':', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax4, time4, roll_bdy5,   '-', 'Color', [0.1 0.1 0.1], 'LineWidth', 2);
set(gca,"FontSize", 11)
xlabel('Time (s)', 'FontSize', 12); ylabel('Body Roll (rad)', 'FontSize', 12);
hold off

ax5 = nexttile;
plot(ax5, time1, pitch_bdy1,  '-.', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax5, time2, pitch_bdy2,   '--', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax5, time3, pitch_bdy3,   '-', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax5, time4, pitch_bdy5,   ':', 'Color', [0 0 0], 'LineWidth', 2);
hold on
plot(ax5, time4, pitch_bdy5,   '-', 'Color', [0.1 0.1 0.1], 'LineWidth', 2);
set(gca,"FontSize", 11)
xlabel('Time (s)', 'FontSize', 12); ylabel('Body Pitch (rad)', 'FontSize', 12);
hold off