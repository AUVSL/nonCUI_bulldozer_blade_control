% the height of a trapizod between its sides H3 and H4 centered around the base of length B1
function Hx = Hx(yb, H3, H4, B1)
    Hx = (H3 - H4)/B1 * yb + (H3 + H4)/2;
end