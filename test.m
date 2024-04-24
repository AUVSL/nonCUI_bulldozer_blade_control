function v_dot = vehicle_dynamics(tau, Rl, Fy, Mr, Fb, Mb, q, q_dot, x_ICR, x_ICR_dot, v, new_bld_ang, vd_parmas)
    % calculate and return the accleration of the vehicle from the equations of motion
    a = q(4); B = q(5); g = q(6); Bd = q_dot(5); gd = q_dot(6); sa = sin(a);
    m = vd_parmas(1); b = vd_parmas(2); l = vd_parmas(3); r = vd_parmas(4); 
    grav = vd_parmas(5); 
    ca = cos(a); sB = sin(B); cB = cos(B); sg = sin(g); cg = cos(g);
    
    ab = new_bld_ang(1); Bb = new_bld_ang(2); gb = new_bld_ang(3);
    cab = cos(ab);  sab = sin(ab); cBb = cos(Bb);  
    sBb = sin(Bb); cgb = cos(gb);  sgb = sin(gb); 

    % rotation matrix from blade to local body frame (R is orthonormal -> R^-1 = R^T)
    R_blade = [cBb*cgb, sab*sBb*cgb - cab*sgb, cab*sBb*cgb + sab*sgb;
               cBb*sgb, sab*sBb*sgb + cab*cgb, cab*sBb*sgb - sab*cgb;
                  -sBb,            sab*cBb,            cab*cBb];
    
    % moment of inertia for a plate around center
    I = 1/12 * m * (b^2 + l^2);
    
    % rotation matrix from local to global (R is orthonormal -> R^-1 = R^T)
    R_lg = [cB*cg, sa*sB*cg - ca*sg, ca*sB*cg + sa*sg;
            cB*sg, sa*sB*sg + ca*cg, ca*sB*sg - sa*cg;
              -sB,            sa*cB,            ca*cB];
         
    % rotation matrix from local to global of local x
    R_lg_x = [cB*cg;
              cB*sg;
               -sB];
   
    % null space of A matrix
    S = [       R_lg_x, [x_ICR*cB*sg; -x_ICR*cB*cg; 0];
         zeros([3, 1]),                   [ 0; 0; 1]];

    % forward track forces
     B  = [R_lg_x, R_lg_x; 0, 0; 0, 0; -ca*cB*b/2, ca*cB*b/2] ./ r;  
     
    %resistive_forces_and_moments
    Ct = [R_lg, zeros(3); zeros(3), R_lg] * [Rl(1) + Rl(2); -Fy; 0; 0; 0; Mr + (Rl(2) - Rl(1))*b/2];
    Cb = [R_blade, zeros(3); zeros(3), R_blade] * [Fb; 0; 0; 0; 0; Mb];
    
    C = Ct + Cb;
   
    % matrix relating to the mass and moment of inertia to the linear and angular kinetic energy
    M = [m, 0, 0, 0, 0, 0;
         0, m, 0, 0, 0, 0;
         0, 0, m, 0, 0, 0;
         0, 0, 0, I, 0, 0;
         0, 0, 0, 0, I, 0;
         0, 0, 0, 0, 0, I];
                         
    % matrix relating to the graviational potential energy
    P = [0; 0; m*grav; 0; 0; 0];
   
    % derivative of the S matrix
     Sd = [-sB*Bd*cg - cB*sg*gd,  x_ICR_dot*cB*sg - x_ICR*sB*Bd*sg + x_ICR*cB*cg*gd;
           -sB*Bd*sg + cB*cg*gd, -x_ICR_dot*cB*cg + x_ICR*sB*Bd*cg + x_ICR*cB*sg*gd;
                         -cB*Bd,                                                  0;
            zeros([3, 2])];

    % tranformation of matrixes to get v_dot
    Bt = S.' * B;
    Ct = S.' * C;
    Pt = S.' * P;
    Mt = S.' * M * S;
    Et = S.' * M * Sd;
   
    % matrix multiplication by inverse Mt to get v_dot
    v_dot = Mt \ (Bt * tau - Et * v - Ct - Pt);
end