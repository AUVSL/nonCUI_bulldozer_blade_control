clear all; clc
% Example angles
% A = 0.3;
% B = -0.6;
% C = 1.1;

% x_ICR = 0.4;


% Symbolic variables
syms A B C x_ICR real

sA = sin(0); cA = cos(0);
sB = sin(0); cB = cos(0); tB = sB/cB;
sC = sin(C); cC = cos(C);

% Rotation matrix R_lg (from your figure)
R_gl = [ ...
    cB*cC,      sA*sB*cC - cA*sC,   cA*sB*cC + sA*sC;
    cB*sC,      sA*sB*sC + cA*cC,   cA*sB*sC - sA*cC;
   -sB,         sA*cB,              cA*cB ];
R_lg = R_gl';

% Extract columns
R_lgx = R_lg(:,1);
R_lgy = R_lg(:,2);
R_lgz = R_lg(:,3);

% Numerical J_gl
J_gl = [1, sA*tB, cA *tB;
        0,    cA,   -sA;
        0, sA/cB, cA/cB ];


J_lg = [ ...
    1,  0,     -sB;
    0,  cA,     sA*cB;
    0, -sA,     cA*cB ];

% y-row
J_glz = J_gl(3,:);
J_lgz  = J_lg(:,3);

Y = J_lgz
S = [R_lgx,     -x_ICR*R_lgy;
     zeros(3,1),      J_lgz];

A = [R_lgy.' x_ICR*J_glz];
% A*S

AS = simplify(A * S)

