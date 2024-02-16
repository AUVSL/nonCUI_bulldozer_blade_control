function out = velSat(in1, in2, saturation)
    vel = sqrt(in1*in1 +  in2*in2);
    if vel > saturation;
        out = in1/vel * saturation;
    else
        out = in1;
    end
end