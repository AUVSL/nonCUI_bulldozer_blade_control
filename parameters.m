% physical contstants
grav       = 9.81; % gravity (m/s^2)
pathlength = 10;   % a length used to trigger the stop condition

% variables specific to the John Deere 650K LGP bulldozer with a standard blade
l = 2.337;       % track length (m)
b = 1.75;        % track gauge (m) - the distance between the center of the tracks    
r = 0.4;         % radius of drive wheel (m)
B1 = 2.921;      % dozer=s blade width (m)
H  = 0.955;      % dozer blade height (m)
X = 0.01;        % dozer blade thickness (m)
L = 1.4;         % dozer blade yaw arm length (m)
m = 9355;        % mass of vehicle (kg)
mu_l = 0.1;      % coefficient of longitudinal resistance (front/back of dozer)
mu_t = 0.9;      % coefficient of lateral resistance (sides of dozer)
mu_ss = 0.5;     % friction coefficient between soil and soil
mu_sb = 0.05;    % friction coefficient between soil and blade
kb = 0.04;       % cutting resistance per unit area after blade pressed into the soil (Mpa)
ky = 0.04;       % cutting resistance per unit area (Mpa)
alpha0_deg = 38; % natural slope angle of soil (degrees)

if(soil == 0)    
    gamma_g = 1480 * 9.81; % gravity per cubic meter (N/m^3)
    km = 0.94;       % fullness degree coefficient of soil
    ks = 1.06;       % loose degree coefficient of soil
    type = 0.2;
    alpha_deg = 0; % yaw angle of slope (degrees)
    beta_deg  = 0; % pitch angle of slope (degrees)
    gamma_deg = 0; % roll angle of the slope (degrees)
else
    gamma_g = 1270 * 9.81; % gravity per cubic meter (N/m^3)
    km = 0.90;       % fullness degree coefficient of soil
    ks = 1.1;       % loose degree coefficient of soil
    type = 0.9;
    alpha_deg = 0; % yaw angle of slope (degrees)
    beta_deg  = 0; % pitch angle of slope (degrees)
    gamma_deg = 0; % roll angle of the slope (degrees)
end

% convert from deg to radians
alpha0 = pi/180 * alpha0_deg;
alpha = pi/180 * alpha_deg;
beta = pi/180 * beta_deg;
gamma = pi/180 * gamma_deg;
    