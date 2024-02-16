function Dx = Dx(x, alpha_0, H3, H4, B1) 
    Dx = Hx(x, H3, H4, B1) * cot(alpha_0);
end