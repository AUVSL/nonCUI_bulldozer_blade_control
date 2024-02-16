function Hx = Hx(x, H3, H4, B1)
    Hx = (H4 - H3)/B1 * x + H3;
end