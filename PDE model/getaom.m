function AOM = getaom(n, L)

h = L / n;
e = ones(n, 1);
AOM = spdiags([-e 0*e e], -1:1, n, n);
%boundary condition
AOM(1, 2) = 0; 
AOM(n, n - 1) = 0;
AOM = AOM / 2 / h;

