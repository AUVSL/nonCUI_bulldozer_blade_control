close all; clear all; clc; format long
addpath(genpath('C:\Users\srd2\Code\nonCUI_bulldozer_blade_control\controllers'));
addpath(genpath('C:\Users\srd2\Code\nonCUI_bulldozer_blade_control\helper_functions'));

% physical contstants
grav       = 9.81;     % gravity (m/s^2)
pathlength = 10; % a length used to trigger the stop condition

% variables specific to the Cat D3, LGP 30 in track
b = 1.7;   % track gauge (m) - the distance between the center of the tracks    
w = 0.3;   % shoe width  (m)
l = 2;     % track length (m)
m = 1450;  % mass of vehicle (kg)
r = 0.3;   % radius of drive wheel (m)

% physical contstants
B1 = 3.54;   % dozer’s blade width (m)
H  = 1.58;   % dozer blade height (m)

% the set of inputs to the simulation
padding = 0;       % padding since simulink drops the first value of input matrices
q_dot   = [0; 0; 0; 0; 0; 0]; % a matrix the global [X_dot; Y_dot; theta_dot]
q       = [0; 0; 0; 0; 0; 0]; % a matrix the global [X; Y; theta]
x_ICR   = 0;          % local x coordinate of the Instntanous Center of Rotation
v       = [0; 0];     % the theoretical velocities of left and right tracks
tau     = [80000; 80000]; % drive wheel torques for the left and right tracks
bld_ang = [0; 0; 0]; % angle of blade about it local x, y, and z-axis

% store the simulation inputs in a single array since the simulink simin
% block only accepts a single variable
simin = [padding, tau(1), tau(2), q_dot(1), q_dot(2), q_dot(3), ...
    q_dot(4), q_dot(5), q_dot(6), q(1), q(2), q(3), q(4), q(5), ...
    q(6), x_ICR, v(1), v(2), bld_ang(1), bld_ang(2), bld_ang(3)];
initial_states = [tau(1); tau(2); q_dot(1); q_dot(2); q_dot(3); ...
    q_dot(4); q_dot(5); q_dot(6); q(1); q(2); q(3); q(4); q(5); ...
    q(6); x_ICR; v(1); v(2); bld_ang(1); bld_ang(2); bld_ang(3)];
initial_v = v;
initial_q = q;

% run the simulation
out = sim('simulation_3d').output.data;

time        = out(:,1, :);
x_pos       = out(:,2, :);
y_pos       = out(:,3, :);
yaw_error   = out(:,4, :);
pitch_error = out(:,5, :);
roll_error  = out(:,6, :);

time        = reshape(time,        length(time)        ,1);
x_pos       = reshape(x_pos,       length(x_pos)       ,1);
y_pos       = reshape(y_pos,       length(y_pos)       ,1);
yaw_error   = reshape(yaw_error,   length(yaw_error)   ,1);
pitch_error = reshape(pitch_error, length(pitch_error) ,1);
roll_error  = reshape(roll_error,  length(roll_error)  ,1);

% plot(time, yaw_error, 'Color', [0 0 0], 'LineWidth', 2);
% xlabel('Time (s)'); ylabel('Yaw Error (rad)');
% [a_rmse, a_me] = courseErrors(yaw_error)
% hold off; figure;

plot(time, pitch_error, 'Color', [0 0 0], 'LineWidth', 2);
xlabel('Time (s)'); ylabel('Pitch Error (rad)');
[b_rmse, b_me] = courseErrors(pitch_error)
hold off; figure;

% plot(time, roll_error, 'Color', [0 0 0], 'LineWidth', 2);
% xlabel('Time (s)'); ylabel('Roll Error (rad)');
% [g_rmse, g_me] = courseErrors(roll_error)