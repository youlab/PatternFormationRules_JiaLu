function DOM = diffusion1Dx_NL(D, n, dr, dt)
    

e = ones(n, 1);

% -------- 2nd derivative term --------
M = spdiags([e -2*e e], -1:1, n, n);
M(1, 2) = 2;
M(n, n - 1) = 2;
M = D * M / (dr ^ 2);
DOM1 = speye(n) - M * dt;

% -------- (1/r)d_r --------
M = spdiags([-e e], [-1, 1], n, n);
rad_vec = linspace(0, 1, n)' + 1e-5; % r
DOM2 = - D * dt / dr / 2 * bsxfun(@rdivide, M, rad_vec);

DOM2(1,:) = 0;       % Eliminate first row becuse of symmetry -> term vanishes
DOM2(n,:) = 0;  


DOM = DOM1 + DOM2;
