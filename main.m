close all; format longG; clc; clear
project_dir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(project_dir, 'controllers')));
addpath(genpath(fullfile(project_dir, 'sim_helper_functions')));
addpath(genpath(fullfile(project_dir, 'paper_preperation')));

% Select one of the proposed-fuzzy cases reported in Table 8 of the paper.
%   1: compact soil with observer noise
%   2: loose soil with observer noise
%   3: compact soil without observer noise
case_id = 1;
controllerIndex1234 = 4; % 4 selects the proposed fuzzy controller

switch case_id
    case 1
        soil               = 0.1;
        noise_power        = 2e-7;
        desired_depth_m    = -0.03;
        desired_angle_rad  = -0.005;
        surface_angle_rad  =  0.005;
    case 2
        soil               = 0.9;
        noise_power        = 2e-7;
        desired_depth_m    = -0.04;
        desired_angle_rad  = -0.003;
        surface_angle_rad  =  0.003;
    case 3
        soil               = 0.1;
        noise_power        = 0;
        desired_depth_m    = -0.03;
        desired_angle_rad  = -0.005;
        surface_angle_rad  =  0.005;
    otherwise
        error('case_id must be 1, 2, or 3.');
end

run("parameters.m") % run file with params

padding = 0; % padding since simulink drops the first value of input matrices

% Control variables. The model currently obtains the depth command from
% blade_height.fis using soil; desired_depth is retained for reference.
desired_depth = [padding, desired_depth_m];
desired_abg   = [desired_angle_rad, 1, desired_angle_rad]; % [roll (rad.), control pitch (1) or not (0), yaw (rad.)]
surface_abg   = [surface_angle_rad, 0, surface_angle_rad]; % the roll, pitch, yaw of the surface (radians)
bld_ang       = [0.0; 0.0; 0.0];     % angle of blade about it local x, y, and z-axis
F_track       = [60000; 60000];      % drive wheel torques for the left and right tracks

% initial State variables
% an array of the global body [X; Y; Z; roll; pitch; yaw] 
q     = [0; 0; 0; surface_abg(1); surface_abg(2); surface_abg(3)];
% an array of the global body [X_vel; Y_vel; Z_vel; roll_vel; pitch_vel; yaw_vel] 
q_dot = [0; 0; 0; 0; 0; 0]; 
x_ICR = 0;      % local body x coordinate of the Instntanous Center of Rotation
v     = [0; 0]; % the velocities of left and right tracks

% store the simulation inputs in a single array since the simulink simin
% block only accepts a single variable

simin          = [padding, F_track(1), F_track(2), q_dot(1), q_dot(2), q_dot(3), ...
                  q_dot(4), q_dot(5), q_dot(6),q(1), q(2), q(3), q(4),   ...
                  q(5), q(6), x_ICR, v(1), v(2), bld_ang(1), bld_ang(2), bld_ang(3)];
initial_states = [F_track(1), F_track(2), q_dot(1), q_dot(2), q_dot(3), q_dot(4), ...
                   q_dot(5), q_dot(6),q(1), q(2), q(3), q(4), q(5), q(6), ...
                   x_ICR, v(1), v(2), bld_ang(1), bld_ang(2), bld_ang(3)]';
desired_angles = [padding, desired_abg];
bt_params      = [padding, B1, H, L, b, l, r, m, grav, velocity_limit,   ...
                  fill_distance, mu_t, mu_l, mu_ss, kb, gamma_g, beta0, surface_abg];
vd_params      = [padding,  m, h, b, l, r, grav];
v_limit        = [velocity_limit, turn_vel_limit];
initial_v      = v; % needed for the track acceleration integrator
initial_q      = q; % needed for the global body velocity integrator

% run the simulation
out12 = sim('simulation_3d').output.data;

% plots and errors
[rmse_r, me_r, rmse_d, me_d, rmse_y, me_y] = errors_and_plots(out12);
[rmse_r, me_r, rmse_d, me_d, rmse_y, me_y]*1000
