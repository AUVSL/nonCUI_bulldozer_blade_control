close all; format long; clc;
addpath(genpath('C:\Users\srd2\Code\nonCUI_bulldozer_blade_control\controllers'));
addpath(genpath('C:\Users\srd2\Code\nonCUI_bulldozer_blade_control\sim_helper_functions'));
addpath(genpath('C:\Users\srd2\Code\nonCUI_bulldozer_blade_control\paper_prep'));

% load in params
soil        = 1;           % select the soil paramters (0 compact, 1 loose)
% noise_power = 0.0;
run("parameters.m") % run file with params

% control varaibles
desired_abg = [-0.01, 1, -0.01]; % [roll (rad.), control pitch (1) or not (0), yaw (rad.)]
surface_abg = [ 0.01, 0,  0.01]; % the roll, pitch, yaw of the surface (radians)
bld_ang     = [0.0; 0.0; 0.0];     % angle of blade about it local x, y, and z-axis
tau         = [60000; 60000];      % drive wheel torques for the left and right tracks

% initial State variables
% an array of the global body [X; Y; Z; roll; pitch; yaw] 
q     = [0; 0; 0; surface_abg(1); surface_abg(2); surface_abg(3)];
% an array of the global body [X_vel; Y_vel; Z_vel; roll_vel; pitch_vel; yaw_vel] 
q_dot = [0; 0; 0; 0; 0; 0]; 
x_ICR = 0;      % local body x coordinate of the Instntanous Center of Rotation
v     = [0; 0]; % the velocities of left and right tracks

% store the simulation inputs in a single array since the simulink simin
% block only accepts a single variable
padding = 0; % padding since simulink drops the first value of input matrices
simin          = [padding, tau(1), tau(2), q_dot(1), q_dot(2), q_dot(3), ...
                  q_dot(4), q_dot(5), q_dot(6),q(1), q(2), q(3), q(4),   ...
                  q(5), q(6), x_ICR, v(1), v(2), bld_ang(1), bld_ang(2), bld_ang(3)];
initial_states = [tau(1), tau(2), q_dot(1), q_dot(2), q_dot(3), q_dot(4), ...
                   q_dot(5), q_dot(6),q(1), q(2), q(3), q(4), q(5), q(6), ...
                   x_ICR, v(1), v(2), bld_ang(1), bld_ang(2), bld_ang(3)]';
desired_angles = [padding, desired_abg];
bt_params      = [padding, B1, H, L, b, l, r, m, grav, velocity_limit,   ...
                  fill_distance, mu_t, mu_l, mu_ss, kb, gamma_g, beta0, surface_abg];
vd_params      = [padding,  m, h, b, l, r, grav];
v_limit        = [velocity_limit, turn_vel_limit];
initial_v      = v; % needed for the track acceleration integrator
initial_q      = q; % needed for the global body velocity integrator

controllerIndex1234 = 4;

% run the simulation
out12 = sim('simulation_3d').output.data;

% plots and errorserrors
[rmse_r, me_r, rmse_d, me_d, rmse_y, me_y] = errors_and_plots(out12);
[rmse_r, rmse_d, rmse_y]*1000