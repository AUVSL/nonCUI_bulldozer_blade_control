function Dx = Dx(yb, beta_0, H3, H4, B1) 
    Dx = Hx(yb, H3, H4, B1) * cot(beta_0);
end