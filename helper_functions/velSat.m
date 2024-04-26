function out = velSat(in1, saturation)
    if in1 > saturation
        out =  saturation;
    elseif in1 < -saturation
        out =  -saturation;
    else
        out = in1;
    end
end