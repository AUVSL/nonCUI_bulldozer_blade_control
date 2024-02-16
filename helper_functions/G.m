function out = G(F, f, dx)
    % determine whether the forward force overcomes static fricitions
    
    if dx ~= 0
        out = f*sign(dx); % moving so already overcame static friction
    elseif ((dx == 0) && (abs(F)<= f)) 
        out = F;           % not moving and force is LESS than static friction
    else 
        out = f*sign(F);  % not moving and force is MORE than static friction
    end
end