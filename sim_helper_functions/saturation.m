% saturation of a value given a min/max limit
function out = saturation(in, limit)
    if in > limit
        out = limit;
    elseif in < -limit
        out = -limit;
    else
        out = in;
    end
end