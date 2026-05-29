function v_dot = vehicle_dynamics(F_track, Rl, Fy, Mr, Fb, Mb, q, q_dot, x_ICR, x_ICR_dot, v, new_bld_ang, vd_parmas)
    % calculate and return the accleration of the vehicle from the equations of motion
    
    % unpack the body angles (roll, pitch, yaw)
    a = q(4); B = q(5); g = q(6); 
    
    % unpack the body velocities  (roll_vel, pitch_vel, yaw_vel)
    Ad = q_dot(4); Bd = q_dot(5); Gd = q_dot(6); 
    
    % unpack function parameters
    m = vd_parmas(1); h = vd_parmas(2); b = vd_parmas(3); l = vd_parmas(4); 
    r = vd_parmas(5); grav = vd_parmas(6); 
 
    sa = sin(a); ca = cos(a); sB = sin(B); cB = cos(B); sg = sin(g); cg = cos(g);
    tB = tan(B);
    
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

   J_lg_z = [ ca * tB;
              -sa;
             ca / cB];

    % null space of A matrix
        S = [       R_lg_x, R_lg_y * -x_ICR;
             zeros([3, 1]),          J_lg_z];

    % forward track forces
    B  = [    R_lg_x,    R_lg_x; 
          -R_lg_z*b/2, R_lg_z*b/2];  
     
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


    S_11 = -sB * cg * Bd - R_lg(2, 1) * Gd;
    S_21 = -sB * sg * Bd + R_lg(1, 1) * Gd;
    S_31 = -cB * Bd;

    S_12 = -x_ICR * ( R_lg(1, 3) * Ad + R_lg(3, 2) * cg * Bd - R_lg(2, 2) * Gd) - x_ICR_dot * R_lg(1, 2);
    S_22 = -x_ICR * ( R_lg(2, 3) * Ad + R_lg(3, 2) * sg * Bd + R_lg(1, 2) * Gd) - x_ICR_dot * R_lg(2, 2);
    S_32 = -x_ICR * ( R_lg(3, 3) * Ad - sa * sB * Bd)                           - x_ICR_dot * R_lg(3, 2);

    S_42 = -sa * tB * Ad + ca / (cB^2) * Bd;
    S_52 = -ca * Ad;
    S_62 = -sa / cB * Ad + ca * tB / cB * Bd;
    
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