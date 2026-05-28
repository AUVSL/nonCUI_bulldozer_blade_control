function [F_track, q_dot, q, x_ICR, v, bld_ang, stop] = simulation_stopping_and_state_loading(in, simin, stop_distance)
    persistent start;
    
    % initalization condition
    if isempty(start)
        in = simin;
        start = 0;
    end
    
    % bring blade angles into their controller domain, [-pi, pi]
    for i = 18:20
        if((in(i) < -pi) || (pi < in(i)))
            in(i) = mod(in(i)+pi, 2*pi) - pi;
        end
    end

    % a stop flag where nonzero values trigger the stop block
    stop = 0;
    if abs(in(9)) + abs(in(10)) > stop_distance
        stop = 1;
    end
    
    % update states
    F_track = [in(1); in(2)]; % drive wheel torques for the left and right tracks
    % an array of the global body [X_vel; Y_vel; Z_vel; roll_vel; pitch_vel; yaw_vel] 
    q_dot   = [in(3);  in(4);  in(5); in(6); in(7);   in(8)];
    % an array of the global body [X; Y; Z; roll; pitch; yaw] 
    q       = [in(9); in(10); in(11); in(12); in(13); in(14)];
    x_ICR   = in(15);                   % local body x coordinate of the Instntanous Center of Rotation  
    v       = [in(16); in(17)];         % the velocities of left and right tracks
    bld_ang = [in(18); in(19); in(20)]; %the orientation of the blade
end