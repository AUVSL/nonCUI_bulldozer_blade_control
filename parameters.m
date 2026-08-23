% parameters specific to the John Deere 650K LGP bulldozer with a standard blade
h  = 2.762;  % vehicle height (m)
l  = 2.349;  % track length (m)
w  = 0.7112; % track width (m)
b  = 1.75;   % track gauge (m) - the distance between the center of the tracks    
r  = 0.4;    % radius of drive wheel (m)
B1 = 2.921;  % dozers blade width (m)
H  = 0.955;  % dozer blade height (m)
L  = 1.2;    % dozer blade yaw arm length (m)
m  = 10156;  % mass of vehicle (kg)

% parameters specific to sandy-loam soil
mu_l      = 0.1;      % coefficient of longitudinal resistance (front/back of dozer)
mu_t      = 0.9;      % coefficient of lateral resistance (sides of dozer)
mu_ss     = 0.5;      % friction coefficient between soil and soil
kb        = 0.734*(10^6);% cutting resistance per unit area (Pa)
beta0_deg = 38;       % natural slope angle of soil (degrees)
c         = 13000;    % cohesion of soil (Pa)

% miscellaneous parameters
grav                      = 9.81;  % gravity (m/s^2)
stop_distance             = 0.3;   % a length used to trigger the stop condition
gain                      = 1/40;  % gain on the pid reprsenting the hydrolics    
velocity_limit            = 2.222; % the maximum speed the dozer can hit (m/s)
fill_distance             = 8;     % the distance traveled required to fill the pile (m)
if soil < 0.5
    gamma_g = 1640 * grav; % compact-soil weight per cubic meter (N/m^3)
else
    gamma_g = 1480 * grav; % loose-soil weight per cubic meter (N/m^3)
end
dt                        = 0.001; % time step (s)
stop_time                 = 2; %simulation stop time (s)

derivative_filter_samples = 1;

% controller gains
KpP = -3.772;


%KiPI = -4.0; % 1st round 
KpPI = -3.772; % 2nd round
KiPI  = -8.92;

KpCong = 85.24;
KiCong = 7.732;
KdCong = 7.732;
KinputCong = 1;

KpProp = -18.028;
KiProp = 0;
KdProp = 0;
KinputProp = 1;

turn_vel_limit = 2*velocity_limit / b;

% convert from deg to radians
beta0 = pi/180 * beta0_deg;

% determine max torque commanded based on what the soil can support
max_torque = c + m*grav / (w*l) * tan(beta0);


