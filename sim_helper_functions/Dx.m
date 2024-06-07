% the depth of a of a trianglular soil pile with a trapoizdal face and 
% a soil pile angle beta_0
function Dx = Dx(yb, beta_0, H3, H4, B1) 
    Dx = Hx(yb, H3, H4, B1) * cot(beta_0);
end

% the height of a trapizod between its sides H3 and H4 centered around the base of length B1
function Hx = Hx(yb, H3, H4, B1)
    Hx = (-H3 + H4)/B1 * yb + (H3 + H4)/2;
end