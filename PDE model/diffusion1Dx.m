function DOM = diffusion1Dx_YB(D, n, dr, dt)
    

e = ones(n, 1);


M = spdiags([e -2*e e], -1:1, n, n);
M(1, 2) = 2;
M(n, n - 1) = 2;
M = D * M / (dr ^ 2);


M(1,:) = 2*M(1,:);
DOM1 = speye(n) - M * dt;

M = spdiags([-e e], [-1, 1], n, n);
rad_vec = linspace(0, 1, n)'; % r
DOM2 = - D * dt / dr / 2 * bsxfun(@rdivide, M, rad_vec);


DOM2(n, n - 1) = 0;
DOM2(1,:) = 0;% Eliminate first row due to singularity 


DOM = DOM1 + DOM2;
