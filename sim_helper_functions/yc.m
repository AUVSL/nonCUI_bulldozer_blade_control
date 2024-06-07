% centroid of a trapizoid with the xy axis at the center of the base, i.e.
% -B1/2, were the sides are D1 and D2 and the base is B1
function ycx = yc(D1, D2, B1)
    ycx = (D1+2*D2)/(3*(D1+D2))*B1 - B1/2;
end