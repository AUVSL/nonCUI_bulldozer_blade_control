clc; close all
time1        = reshape(out1(:, 1,:), length(out1(:, 1,:)), 1);
roll_error1  = reshape(out1(:, 8,:), length(out1(:, 8,:)), 1);
depth_error1 = reshape(out1(:, 9,:), length(out1(:, 9,:)), 1);
yaw_error1   = reshape(out1(:,10,:), length(out1(:,10,:)), 1);

time2        = reshape(out2(:, 1,:), length(out2(:, 1,:)), 1);
roll_error2  = reshape(out2(:, 8,:), length(out2(:, 8,:)), 1);
depth_error2 = reshape(out2(:, 9,:), length(out2(:, 9,:)), 1);
yaw_error2   = reshape(out2(:,10,:), length(out2(:,10,:)), 1);

time3        = reshape(out3(:, 1,:), length(out3(:, 1,:)), 1);
roll_error3  = reshape(out3(:, 8,:), length(out3(:, 8,:)), 1);
depth_error3 = reshape(out3(:, 9,:), length(out3(:, 9,:)), 1);
yaw_error3   = reshape(out3(:,10,:), length(out3(:,10,:)), 1);

time4        = reshape(out4(:, 1,:), length(out4(:, 1,:)), 1);
roll_error4  = reshape(out4(:, 8,:), length(out4(:, 8,:)), 1);
depth_error4 = reshape(out4(:, 9,:), length(out4(:, 9,:)), 1);
yaw_error4   = reshape(out4(:,10,:), length(out4(:,10,:)), 1);

% --- create a figure with the final size you want ---
fig = figure('Units','inches','Position',[1 1 7.4 5], 'Color','w');

% Tight layout: removes extra gaps around tiles
tiledlayout(fig,3,1, 'Padding','compact')

% ---------- Tile 1 (Depth error) ----------
ax1 = nexttile;
plot(ax1, time1, depth_error1, '-.', 'Color', [0.0 0.0 0.0], 'LineWidth', 2);  hold on
plot(ax1, time2, depth_error2,  ':', 'Color', [0.0 0.0 0.0], 'LineWidth', 2);
plot(ax1, time3, depth_error3, '--', 'Color', [0 0 0], 'LineWidth', 2);
plot(ax1, time4, depth_error4,  '-', 'Color', [0.4 0.4 0.4], 'LineWidth', 2);
set(ax1,"FontSize", 12); ylabel('Depth (m)', 'FontSize', 12);

legend({'P', 'PI', 'Fzy PID', 'Fzy (Prop.)'}, 'Location','northoutside', ...
    'Orientation', 'horizontal', 'NumColumns', 4, FontSize=12)

% ---------- Tile 2 (Roll error) ----------
ax2 = nexttile;
plot(ax2, time1, roll_error1, '-.', 'Color', [0.0 0.0 0.0], 'LineWidth', 2); hold on
plot(ax2, time2, roll_error2,  ':', 'Color', [0.0 0.0 0.0], 'LineWidth', 2);
plot(ax2, time3, roll_error3, '--', 'Color', [0 0 0], 'LineWidth', 2);
plot(ax2, time4, roll_error4,  '-', 'Color', [0.4 0.4 0.4], 'LineWidth', 2);
set(ax2,"FontSize", 12); ylabel('Roll Error (rad)', 'FontSize',12);

% ---------- Tile 3 (Yaw error) ----------
ax3 = nexttile;
plot(ax3, time1, yaw_error1, '-.', 'Color', [0.0 0.0 0.0], 'LineWidth', 2); hold on
plot(ax3, time2, yaw_error2,  ':', 'Color', [0.0 0.0 0.0], 'LineWidth', 2);
plot(ax3, time3, yaw_error3, '--', 'Color', [0 0 0], 'LineWidth', 2);
plot(ax3, time4, yaw_error4,  '-', 'Color', [0.4 0.4 0.4], 'LineWidth', 2);
set(ax3,"FontSize", 12)
ylabel('Yaw Error (rad)', 'FontSize', 12);

% Only bottom tile gets the x‑label; for others hide tick labels to save space
set([ax1 ax2 ax3],'XTickLabel',[]);
xlabel('Time (s)', 'FontSize', 12);