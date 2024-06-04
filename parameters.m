% physical contstants
grav       = 9.81; % gravity (m/s^2)
pathlength = 10;   % a length used to trigger the stop condition

% variables specific to the John Deere 650K LGP bulldozer with a standard blade
h = 2.762;       % vehicle height (m)
l = 2.349;       % track length (m)
w = 0.7112;      % track width (m)
b = 1.75;        % track gauge (m) - the distance between the center of the tracks    
r = 0.4;         % radius of drive wheel (m)
B1 = 2.921;      % dozers blade width (m)
H  = 0.955;      % dozer blade height (m)
L = 1.4;         % dozer blade yaw arm length (m)
m = 10156;       % mass of vehicle (kg)
mu_l = 0.1;      % coefficient of longitudinal resistance (front/back of dozer)
mu_t = 0.9;      % coefficient of lateral resistance (sides of dozer)
mu_ss = 0.5;     % friction coefficient between soil and soil
kb = 40000;      % cutting resistance per unit area (Pa)
beta0_deg = 38;  % natural slope angle of soil (degrees)
c = 13000;       % cohesion of soil (Pa)
 
if(soil == 0)    
    gamma_g = 1480 * 9.81 * 1000; % gravity per cubic meter (N/m^3)
    km = 0.94;       % fullness degree coefficient of soil
    ks = 1.06;       % loose degree coefficient of soil
    type = 0.2;
else
    gamma_g = 1270 * 9.81 * 1000; % gravity per cubic meter (N/m^3)
    km = 0.90;       % fullness degree coefficient of soil
    ks = 1.1;       % loose degree coefficient of soil
    type = 0.9;
end

% convert from deg to radians
beta0 = pi/180 * beta0_deg; 
max_torque = c + m*grav / (w*l) * tan(beta0);