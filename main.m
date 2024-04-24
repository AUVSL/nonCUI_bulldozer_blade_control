close all; clear all; clc; format long
addpath(genpath('C:\Users\srd2\Code\nonCUI_bulldozer_blade_control\controllers'));
addpath(genpath('C:\Users\srd2\Code\nonCUI_bulldozer_blade_control\helper_functions'));

% load in params
soil = 1;           % select the soil paramters (0 compact, 1 loose)
run("parameters.m") % run file with params

% testing varaibles
desired_abg = [0, 0, 0];   % [control pitch (1) or not (0), pitch (rad.), roll (rad.)]
surface_abg = [0, 0, 0.35]; % the yaw, pitch, and roll of the surface (radians)
bld_ang     = [0; 0; 0.35];    % angle of blade about it local x, y, and z-axis
bool_abg    = [0, 0, 1];   % plot yaw, pitch, roll [1 = True, 0 = False]

% the set of inputs to the simulation
padding = 0;       % padding since simulink drops the first value of input matrices
q_dot   = [0; 0; 0; 0; 0; 0]; % a matrix the global [X_dot; Y_dot; theta_dot]
q       = [0; 0; 0; 0; 0; 0]; % a matrix the global [X; Y; theta]
x_ICR   = 0;          % local x coordinate of the Instntanous Center of Rotation
v       = [0; 0];     % the theoretical velocities of left and right tracks
tau     = [56000; 56000]; % drive wheel torques for the left and right tracks

% store the simulation inputs in a single array since the simulink simin
% block only accepts a single variable, padded needed because MATLAB
% deletes the first index for simIn blocks
simin = [padding, tau(1), tau(2), q_dot(1), q_dot(2), q_dot(3), ...
    q_dot(4), q_dot(5), q_dot(6), q(1), q(2), q(3), q(4), q(5), ...
    q(6), x_ICR, v(1), v(2), bld_ang(1), bld_ang(2), bld_ang(3)];
initial_states = [tau(1); tau(2); q_dot(1); q_dot(2); q_dot(3); ...
    q_dot(4); q_dot(5); q_dot(6); q(1); q(2); q(3); q(4); q(5); ...
    q(6); x_ICR; v(1); v(2); bld_ang(1); bld_ang(2); bld_ang(3)];
soil_var = [padding, mu_t, mu_l, mu_ss, mu_sb, kb, km, ks, ky, gamma_g, ...
            beta0, surface_abg];
desired_angles = [padding, desired_abg];
bt_params = [padding, B1, H, X, L, b, l, m, grav];
vd_params = [padding, m, b, l, r, grav];
initial_v = v;
initial_q = q;

% run the simulation
out = sim('simulation_3d').output.data;

% plot errors
plots(bool_abg(1), bool_abg(2), bool_abg(3), out);