function v_dot = vehicle_dynamics(F_track, Rl, Fy, Mr, Fb, Mb, q, q_dot, x_ICR, x_ICR_dot, v, new_bld_ang, vd_parmas)
    % calculate and return the accleration of the vehicle from the equations of motion
    
    % unpack the body angles (roll, pitch, yaw)
    a = q(4); B = q(5); g = q(6); 
    
    % unpack the body velocities  (roll_vel, pitch_vel, yaw_vel)
    Ad = q_dot(4); Bd = q_dot(5); Gd = q_dot(6); 
    
    % unpack function parameters
    m = vd_parmas(1); h = vd_parmas(2); b = vd_parmas(3); l = vd_parmas(4); 
    r = vd_parmas(5); grav = vd_parmas(6); 

    if x_ICR == 0
        x_ICR = realmax;
    end

    sa = sin(a); ca = cos(a); sB = sin(B); cB = cos(B); sg = sin(g); cg = cos(g);
    
    ab = new_bld_ang(1); Bb = new_bld_ang(2); gb = new_bld_ang(3);
    cab = cos(ab);  sab = sin(ab); cBb = cos(Bb);  
    sBb = sin(Bb); cgb = cos(gb);  sgb = sin(gb); 

    % rotation matrix from blade to local body frame (R is orthonormal -> R^-1 = R^T)
    R_blade = [cBb*cgb, sab*sBb*cgb - cab*sgb, cab*sBb*cgb + sab*sgb;
               cBb*sgb, sab*sBb*sgb + cab*cgb, cab*sBb*sgb - sab*cgb;
                  -sBb,               sab*cBb,              cab*cBb];
    
    % moment of inertia for a plate around center
    Ix = 1/12 * m * (b^2 + h^2);
    Iy = 1/12 * m * (h^2 + l^2);
    Iz = 1/12 * m * (b^2 + l^2);
    
    % rotation matrix from local to global (R is orthonormal -> R^-1 = R^T)
    R_lg = [cB*cg, sa*sB*cg - ca*sg, ca*sB*cg + sa*sg;
            cB*sg, sa*sB*sg + ca*cg, ca*sB*sg - sa*cg;
              -sB,            sa*cB,            ca*cB];
         
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

    % forward track forces
    B  = [    R_lg_x,    R_lg_x; 
                   0,         0; 
                   0,         0; 
          -ca*cB*b/2, ca*cB*b/2];  
     
    %resistive_forces_and_moments
    elim = [1, 0, 0, 0, 0, 0;
            0, 1, 0, 0, 0, 0;
            0, 0, 1, 0, 0, 0;
            0, 0, 0, 0, 0, 0;
            0, 0, 0, 0, 0, 0;
            0, 0, 0, 0, 0, 1];
    
    Ct = [Rl(1) + Rl(2); Fy; 0; 0; 0; Mr + (Rl(2) - Rl(1))*b/2];
    Cb = elim * [R_blade, zeros(3); zeros(3), R_blade] * [Fb; 0; 0; 0; 0; Mb];
    C = [R_lg, zeros(3); zeros(3), R_lg] * (Ct + Cb);

    % matrix relating to the mass and moment of inertia to the linear and angular kinetic energy
    M = [m, 0, 0,  0,  0,  0;
         0, m, 0,  0,  0,  0;
         0, 0, m,  0,  0,  0;
         0, 0, 0, Ix,  0,  0;
         0, 0, 0,  0, Iy,  0;
         0, 0, 0,  0,  0, Iz];
                         
    % matrix relating to the graviational potential energy
    P = [0; 0; m*grav; 0; 0; 0];

    S_11 = -sB*cg*Bd - cB*sg*Gd;
    S_21 = (ca*sB*cg + sa*sg)*Ad + sa*cB*cg*Bd - (sa*sB*sg + ca*cg)*Gd;
    S_31 = (ca*sg - sa*sB*cg)*Ad + ca*cB*cg*Bd + (sa*cg - ca*sB*sg)*Gd;
    S_12 = -sB*sg*Bd + cB*cg*Gd;
    S_22 = (ca*sB*sg - sa*cg)*Ad + sa*cB*sg*Bd + (sa*sB*cg - ca*sg)*Gd;

    S_32 = -(sa*sB*sg + ca*cg)*Ad + ca*cB*sg*Bd + (ca*sB*cg + sa*sg)*Gd;
    S_42 = cB*x_ICR^(-1)*Bd - sB*x_ICR^(-2)*x_ICR_dot;
    S_52 = -ca*cB*x_ICR^(-1)*Ad + sa*sB*x_ICR^(-1)*Bd + sa*cB*x_ICR^(-2)*x_ICR_dot;
    S_62 = sa*cB*x_ICR^(-1)*Ad + ca*sB*x_ICR^(-1)*Bd + ca*cB*x_ICR^(-2)*x_ICR_dot;
    
    Sd = [S_11, S_12;
          S_21, S_22;
          S_31, S_32;
             0, S_42;
             0, S_52;
             0, S_62];

    % tranformation of matrixes to get v_dot
    Bt = S.' * B;
    Ct = S.' * C;
    Pt = S.' * P;
    Mt = S.' * M * S;
    Et = S.' * M * Sd;

    % matrix multiplication by inverse Mt to get v_dot
    v_dot = Mt \ (Bt * F_track + Ct - Et * v - Pt);
end