function [FT, Mb] = blade_terrain_interaction(hp, kb, gamma_g, ...
    mu_ss, a_s, beta0, a_b, B1, H, fill_percent)
    % calculate and return the how the ground resistants the blade

    % calcuare the relative roll of the blade to the surface and 
    % the relate soil blade heights
    a_rel = a_s - a_b;
    H1    = B1*tan(abs(a_rel));
    H2    = hp*sec(a_rel);
    H3    = H - H2 + sign(a_rel)*H1/2 - H1/2;
    H4    = H - H2 - sign(a_rel)*H1/2 - H1/2;
    
    % volume of the mound before bulldozing plate
    a = tan(abs(a_rel))^2;
    c = (H3 + H4)/2;
    V = 1/2*cot(beta0)*(1/12*a^2*B1^3 + c^2*B1);
    
    % gravity of the mound before bulldozing plate
    Gt = V * gamma_g * fill_percent;
    
    % soil-cutting resistance (N)
    hyp      = B1*sec(abs(a_rel));
    area_cut = 1/2*B1*H1 + hyp*hp;
    F1       = area_cut * kb;
    
    % pushing resistance of mound before blade (N)
    F2 = Gt * mu_ss;             
    
    % blade pushback forces (N)
    FT = - F1 - F2;
    
    % calculte where the blade forces are acting
    yc1 = yc(Dx(-B1/2, beta0, H3, H4, B1), Dx(B1/2, beta0, H3, H4, B1), B1);
    yc2 = yc(H2, H1+H2, B1);
    
    % soil cutting moment acting on the blade
    Mb = yc1 * F1 + yc2 * F2;
end