%% time and space setups
param.L = 10;
param.N = 1001;
param.tmax = 20;
param.dt = 0.025; 
param.L_pam = 5;

% Model parameters 
param.DC = get_rand(0.5e-3, 12.5e-2, 1, 'log');
param.DN = 0.6; 
param.DA = 0.6;  
param.DB = 0.6;
param.aC = get_rand(0.1, 1, 1, 'linear');
param.aA = get_rand(100, 100000, 1, 'log');
param.aB = 100;
param.aT = get_rand(10, 8000, 1, 'log');
param.aL = get_rand(5, 500, 1, 'log');
param.bN = 155;
param.dA = get_rand(0.001, 0.1, 1, 'log');
param.dB = param.dA;
param.dT = get_rand(3, 300, 1, 'log');
param.dL = get_rand(0.144, 14.4, 1, 'log') ;
param.k1 = 0.400;
param.k2 = 10.800;
param.KN = 20000;
param.KP = 40;
param.KT = 12;
param.KA = 10;
param.KB = 200;
param.alpha = get_rand(1, 5, 1, 'linear');
param.beta = get_rand(2, 2000, 1, 'log');
param.Cmax = 3e5;
param.a = 5;
param.b = 5;
param.m = 2;
param.n = 2; 
param.Kphi = get_rand(1, 10, 1, 'int');
param.l = 2;
param.N0 = get_rand(200000, 5000000, 1, 'linear');

% Calculate nondimensional parameters
param.G1 = param.DC / param.aC / param.L_pam^2;
param.G2 = param.KN / param.N0;
param.G3 = param.DN / param.aC / param.L_pam^2;
param.G4 = param.bN * param.Cmax / param.N0;
param.G5 = param.DA / param.aC / param.L_pam^2;
param.G6 = param.aA * param.Cmax / param.aC / param.KA;
param.G7 = param.KT * param.dT / param.aT;
param.G8 = param.dA / param.aC;
param.G9 = param.DB / param.aC / param.L_pam^2;
param.G10 = param.aB * param.Cmax / param.aC / param.KB;
param.G11 = param.dB / param.aC;
param.G12 = param.dT / param.aC;
param.G13 = param.k1 * param.aL / param.aC / param.dL;
param.G14 = param.k2 * param.KP * param.dT / param.aC / param.aT;
param.G15 = param.dL / param.aC;
param.G16 = param.k1 * param.aT / param.aC / param.dT;
param.G17 = param.k2 * param.KP * param.dL / param.aC / param.aL;
param.G18 = param.k1 * param.aT * param.aL / param.KP / param.aC / param.dT / param.dL;
param.G19 = param.k2 / param.aC;
param.alpha_p = param.alpha * param.aT / param.dT;
param.beta_p = param.beta * param.aL / param.dL;


% Set initial values 
param.Ce0 = get_C0_scale(param, "donut")';
param.Nu0 = ones(param.N, 1); 
param.A0  = zeros(param.N, 1); 
param.B0  = zeros(param.N, 1); 
param.Ly0 = zeros(param.N, 1); 
param.T0  = 0.1 * param.Ce0; 
param.P0  = zeros(param.N, 1); 
param.CFP0  = zeros(param.N, 1); 
param.RFP0  = zeros(param.N, 1); 
