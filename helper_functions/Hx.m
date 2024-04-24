function Hx = Hx(yb, H3, H4, B1)
    Hx = (-H3 + H4)/B1 * yb + (H3 + H4)/2;
end