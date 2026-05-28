function q_dot = v_to_q_dot(q, x_ICR, v)
    % calculate and return the global velocities using a transfromation from the
    % local to global frame
    
    % unpack the body angles (roll, pitch, yaw)
    a = q(4); B = q(5); g = q(6); 
    
    sa = sin(a); ca = cos(a); sB = sin(B); cB = cos(B); sg = sin(g); cg = cos(g);
    
    if abs(x_ICR) < 0.001
        x_ICR = realmax;
    end
     % rotation matrix from local to global of local x, y, and z
        R_lg_x = [cB*cg;
                  cB*sg;
                   -sB];
        R_lg_y = [sa*sB*cg - ca*sg;
                sa*sB*sg + ca*cg;
                           sa*cB];
       R_lg_z = [ca*sB*cg + sa*sg;
                 ca*sB*sg - sa*cg;
                            ca*cB];
    
        % null space of A matrix
        S = [       R_lg_x,               R_lg_y;
             zeros([3, 1]), R_lg_z .* (-1/x_ICR)];

     % the global velocities
     q_dot = S*v;
end