clear all; clc

% Symbolic variables
syms A B C x_ICR real

sA = sin(A); cA = cos(A);
sB = sin(B); cB = cos(B); tB = sB/cB;
sC = sin(C); cC = cos(C);

% Rotation matrix global to local
R_lg = [ ...
    cB*cC,      sA*sB*cC - cA*sC,   cA*sB*cC + sA*sC;
    cB*sC,      sA*sB*sC + cA*cC,   cA*sB*sC - sA*cC;
   -sB,         sA*cB,              cA*cB ];

% Extract columns
R_lgx = R_lg(:,1);
R_lgy = R_lg(:,2);
R_lgz = R_lg(:,3);

J_lg = [1, sA*tB, cA *tB;
        0,    cA,   -sA;
        0, sA/cB, cA/cB ];

J_gl = [ ...
    1,  0,     -sB;
    0,  cA,     sA*cB;
    0, -sA,     cA*cB ];

J_glz = J_gl(3,:);
J_lgz = J_lg(:,3);

S = [R_lgx,     -x_ICR*R_lgy;
     zeros(3,1),      J_lgz];

A = [R_lgy.' x_ICR*J_glz];

AS = simplify(A * S)