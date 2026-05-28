function [new_ang, out] = hydraulics(bld_ang, bld_ang_vel, gain)

    new_roll  = bld_ang(1) + gain * bld_ang_vel(1);
    new_pitch = bld_ang(2) + gain * bld_ang_vel(2);
    new_yaw   = bld_ang(3) + gain * bld_ang_vel(3);
    
    new_ang = [new_roll, new_pitch, new_yaw];
    out = new_ang.';
end