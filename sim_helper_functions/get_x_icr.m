function x_ICR  = get_x_icr(q, q_dot, l)
    % get the x coordinate of the instantanous center of rotation and the
    % slip angle from the local linear and angular velocities
    
    % unpack the body angles (roll, pitch, yaw)
    a = q(4); B = q(5); g = q(6); 
    
    % unpack the body velocities and angular velocities
    X_dot = q_dot(1); Y_dot = q_dot(2); Z_dot = q_dot(3); a_dot = q_dot(4); 
    B_dot = q_dot(5); g_dot = q_dot(6); 
    
    sa = sin(a); ca = cos(a); sB = sin(B); cB = cos(B); sg = sin(g); 
    cg = cos(g);
    
    % calculate the local forward velocity from the vehicle's global X and 
    % Y velocities
	R_gl = [cB*cg, sa*sB*cg - ca*sg, ca*sB*cg + sa*sg
        	cB*sg, sa*sB*sg + ca*cg, ca*sB*sg - sa*cg
        	  -sB,         	  sa*cB,         	ca*cB].';

    J_gl = [1,   0,     -sB
            0,  ca, sa * cB
            0, -sa, ca * cB];
   	 
	% calculate the local forward (dx) and lateral (dy) velocities
	% from the vehicle's global X and Y velocities
	dxyz = R_gl * [X_dot;Y_dot;Z_dot];
	daBg = J_gl * [a_dot;B_dot;g_dot];
    
    % Avoid a divide by infinity error with the if statement
    if abs(daBg(3)) < 0.001
        x_ICR = 0;
    else
        x_ICR = -dxyz(2)/daBg(3);
        x_ICR = max(-l/2, min(x_ICR, l/2));
    end
    
end