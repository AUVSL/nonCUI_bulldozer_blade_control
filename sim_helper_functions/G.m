% the force due to friction
function out = G(F, f, dx)
    % determine whether the forward force overcomes static fricitions
    if (abs(dx) > 1e-10)
        out = -f*sign(dx); % moving so already overcame static friction
    elseif (abs(F)<= f) 
        out = -F;           % not moving and force is LESS than static friction
    else 
        out = -f*sign(F);  % not moving and force is MORE than static friction
    end
end