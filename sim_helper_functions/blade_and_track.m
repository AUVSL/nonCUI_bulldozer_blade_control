function [Rl, Fy, Mr, Fb, Mb] = blade_and_track(F_track, q, q_dot, x_ICR, new_bld_ang, bt_params)
    % calculate the blade and track forces 
    
    % unpack the body angles (roll, pitch, yaw)
    a = q(4); B = q(5); g = q(6); 
    
    % unpack the body velocities and angular velocities
    X_dot = q_dot(1); Y_dot = q_dot(2); Z_dot = q_dot(3); a_dot = q_dot(4); 
    B_dot = q_dot(5); g_dot = q_dot(6);
    
    % unpack function parameters
    B1 = bt_params(1); H = bt_params(2); L = bt_params(3); b = bt_params(4); 
    l = bt_params(5); r = bt_params(6); m = bt_params(7); grav = bt_params(8); 
    velocity_limit = bt_params(9); fill_distance = bt_params(10); 
    mu_t = bt_params(11); mu_l = bt_params(12); mu_ss = bt_params(13); 
    kb = bt_params(14); gamma_g = bt_params(15); beta0 = bt_params(16); a_s = bt_params(17); 
    B_s = bt_params(18); g_s = bt_params(19); 
    
    % unpack the blade angles (roll, pitch, yaw)
    a_b = new_bld_ang(1); B_b = new_bld_ang(2); g_b = new_bld_ang(3);

    sa = sin(a); ca = cos(a); sB = sin(B); cB = cos(B); sg = sin(g); 
    cg = cos(g); 

    % calculate the cutting depth of the blade
    hp = abs(L*sin(B_b));
    
    % calculate the local forward velocity from the vehicle's global X and Y velocities
	R_gl = [cB*cg, sa*sB*cg - ca*sg, ca*sB*cg + sa*sg
        	cB*sg, sa*sB*sg + ca*cg, ca*sB*sg - sa*cg
        	  -sB,         	  sa*cB,         	ca*cB].';
   	 
	% calculate the local forward (dx) and lateral (dy) velocities
	% from the vehicle's global X and Y velocities
	dxyz = R_gl * [X_dot;Y_dot;Z_dot];
	daBg = R_gl * [a_dot;B_dot;g_dot];
    
    % Compute absolute velocities of the centre of the tracks
    vtL = dxyz(1) - b/2 * daBg(3);
    vtR = dxyz(1) + b/2 * daBg(3);
    
    vtL = saturation(vtL, velocity_limit);
    vtR = saturation(vtR, velocity_limit);
    
    % how full the pile will be out some some distance
    fill_percent = sqrt(q(1)^2+q(2)^2+q(3)^2) / fill_distance;

    [Fb, Mb] = blade_terrain_interaction(hp, kb, gamma_g, ...
                      mu_ss, a_s, beta0, a_b, B1, H, fill_percent);
    %left and right track force
    FtL = F_track(1);
    FtR = F_track(2);

    % longitudinal friction distribution and resistance forces
    rl  = mu_l * m*grav/2;       
    RlL = G(FtL, rl, vtL);
    RlR = G(FtR, rl, vtR);
    Rl  = [RlL; RlR];
    
    % lateral friction distribution and resistance force
    fy = mu_t * m*grav/l;
    Fy = -2 * sign(dxyz(2)) * fy * abs(x_ICR);
    
    mr = 2 * fy * ((l^2)/4 - x_ICR^2);      % resistance moment
    M  = ((FtR + RlR) - (FtL + RlL)) * b/2; % turning moment    
    Mr = G(M, mr, daBg(3));                 % moment of turning resistance
end