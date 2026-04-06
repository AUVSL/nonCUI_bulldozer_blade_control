function [errors, plot] = controller_errors(bld_ang, depth, des_ang, L)
    % unpack the blade angles (roll, pitch, yaw)
    roll = bld_ang(1);          pitch = bld_ang(2);         yaw = bld_ang(3);

    % unpack the desired blade angles (roll, pitch, yaw)
    desired_roll = des_ang(1); des_pitch_mult = des_ang(2); desired_yaw = des_ang(3);

    desired_pitch = des_pitch_mult * asin(depth/L);
    
    errors = [roll - desired_roll, pitch - desired_pitch, yaw - desired_yaw];
    
    plot = [errors(1), sin(errors(2))*L, errors(3)];
end