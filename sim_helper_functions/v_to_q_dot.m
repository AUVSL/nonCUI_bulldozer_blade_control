function q_dot = v_to_q_dot(q, x_ICR, v)
    % calculate and return the global velocities using a transfromation from the
    % local to global frame
    
    % unpack the body angles (roll, pitch, yaw)
    a = q(4); B = q(5); g = q(6); 
    
    sa = sin(a); ca = cos(a); sB = sin(B); cB = cos(B); sg = sin(g); cg = cos(g);
    tB = tan(B);
    
     % rotation matrix from local to global of local x, y, and z
        R_lg_x = [cB*cg;
                  cB*sg;
                   -sB];
        R_lg_y = [sa*sB*cg - ca*sg;
                  sa*sB*sg + ca*cg;
                            sa*cB];
       J_lg_z = [ ca * tB;
                      -sa;
                  ca / cB];
    
        % null space of A matrix
        S = [       R_lg_x, R_lg_y * -x_ICR;
             zeros([3, 1]),          J_lg_z];

     % the global velocities
     q_dot = S*v;
end