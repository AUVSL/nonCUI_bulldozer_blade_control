% physical contstants
grav       = 9.81; % gravity (m/s^2)
pathlength = 10;   % a length used to trigger the stop condition

% variables specific to the John Deere 650K LGP bulldozer with a standard blade
l = 2.337;     % track length (m)
b = 1.75;  % track gauge (m) - the distance between the center of the tracks    
r = 0.4;   % radius of drive wheel (m)
B1 = 2.921; % dozer=s blade width (m)
H  = 0.955; % dozer blade height (m)
X = 0.01;  % dozer blade thickness (m)
L = 1.4;  % dozer blade yaw arm length (m)
m = 9355;  % mass of vehicle (kg)

if(soil == 0)
    mu_l = 0.6;  % coefficient of longitudinal resistance (front/back of dozer)
    mu_t = 0.7;  % coefficient of lateral resistance (sides of dozer)
    mu_ss = 1.0; % friction coefficient between soil and soil
    mu_sb = 0.6; % friction coefficient between soil and blade
    kb = 0.00003; % cutting resistance per unit area after blade pressed into the soil (Mpa)
    km = 0.98;    % fullness degree coefficient of soil
    ks = 1.02;    % loose degree coefficient of soil
    ky = 0.00009; % cutting resistance per unit area (Mpa)
    gamma_g = 1900 * 9.81; % gravity per cubic meter (N/m^3)
    alpha0_deg = 30;       % natural slope angle of soil (degrees)
    alpha_deg = 0; % yaw angle of slope (degrees)
    beta_deg = 0;  % pitch angle of slope (degrees)
    gamma_deg = 0; % roll angle of the slope (degrees)
    type = 0.2;
else
    mu_l = 0.6;  % coefficient of longitudinal resistance (front/back of dozer)
    mu_t = 0.8;  % coefficient of lateral resistance (sides of dozer)
    mu_ss = 1.0; % friction coefficient between soil and soil
    mu_sb = 0.5; % friction coefficient between soil and blade
    kb = 0.00001; % cutting resistance per unit area after blade pressed into the soil (Mpa)
    km = 0.94;    % fullness degree coefficient of soil
    ks = 1.06;    % loose degree coefficient of soil
    ky = 0.00006; % cutting resistance per unit area (Mpa)
    gamma_g = 1840 * 9.81; % gravity per cubic meter (N/m^3)
    alpha0_deg = 25;       % natural slope angle of soil (degrees)
    alpha_deg = 0; % yaw angle of slope (degrees)
    beta_deg = 0;  % pitch angle of slope (degrees)
    gamma_deg = 0; % roll angle of the slope (degrees)
    type = 0.9;
end

% convert from deg to radians
alpha0 = pi/180 * alpha0_deg;
alpha = pi/180 * alpha_deg;
beta = pi/180 * beta_deg;
gamma = pi/180 * gamma_deg;
    