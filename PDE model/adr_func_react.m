function out = adr_func_react(t,vec,param,tt)

%% Import parameters used in this function       
N = param.N; % number of grid points
length = param.L; % length of interval
h = length/(N-1);

% need adjustment
Ce = vec(1:N);
Nu = vec(N+1:2*N);
A  = vec(2*N+1:3*N);
B  = vec(3*N+1:4*N);
Ly = vec(4*N+1:5*N);   
T  = vec(5*N+1:6*N);
P  = vec(6*N+1:7*N);
RFP = vec(7*N+1:8*N);
CFP = vec(8*N+1:9*N);

G1 = param.G1; 
G2 = param.G2;
G3 = param.G3;
G4 = param.G4;
G5 = param.G5;
G6 = param.G6;
G7 = param.G7;
G8 = param.G8;
G9 = param.G9;
G10 = param.G10;
G11 = param.G11;
G12 = param.G12;
G13 = param.G13;
G14 = param.G14;
G15 = param.G15;
G16 = param.G16;
G17 = param.G17;
G18 = param.G18;
G19 = param.G19;

m = param.m; 
n = param.n;
l = param.l;
alpha_p =  param.alpha_p;
beta_p =  param.beta_p;
CFP_a = param.a; 
mCh_b = param.b;
a = param.a; 
b = param.b;
Kphi = param.Kphi;


rad_vec = linspace(0, param.L, param.N)'; % radius of each grid point

max_Ce = max(Ce);
if max_Ce > 0
    a = find(Ce / max_Ce >= .9, 1, 'last'); % larger threshold, more uniform expression
    if isempty(a)
        a = 0;
    end
else
    a = 0;
end
h = param.L / (param.N -1)
dist = max((h* a - rad_vec), 0);
GK = (Ce >0) .* Kphi^l ./ (Kphi^l + dist.^l);


META_ACT = meta_act(Ce, Nu, Ly, T, param);

% Cells 
dCedt_r = Ce.*META_ACT;

% Nutrient 
dNudt_r = -G4*Ce.*META_ACT;  

% A 
dAdt_r = G6./(1+P).*T./(T+G7).*GK.*Ce - G8.*A;

% B 
dBdt_r = G10.*GK.*Ce - G11.*B; 

% T7 
dTdt_r = -T.*META_ACT - G12.*T + G12.*T./(T+G7)./(1+P).*GK - G13.*Ly.*T + G14.*P; 

% Lysozyme 
dLydt_r = -Ly.*META_ACT - G15.*Ly + G15.*T./(T+G7).*((A./Ce).^m./(1+(A./Ce).^m) + (B./Ce).^n./(1+(B./Ce).^n)).*GK - G16.*Ly .* T + G17.*P;

%dLydt_r = -Ly.*META_ACT - G15.*Ly + G15.*T./(T+G7).*(A.^m./(1+A.^m) + B.^n./(1+B.^n)).*GK - G16.*Ly .* T + G17.*P;


% P 
dPdt_r = -P.*META_ACT + G18.*Ly.*T - G19.*P;

% CFP (T7 indicator) 
dCFPdt_r = -CFP.*META_ACT  - G12.*CFP + G12.*T./(T+G7)./(1+P).*GK + CFP_a;

% RFP (Lysozyme indicator) 
% dRFPdt_r = -RFP.*META_ACT - G15.*RFP + G15.*T./(T+G7).*(A.^m./(1+A.^m) + B.^n./(1+B.^n)).*GK + mCh_b;
dRFPdt_r = -RFP.*META_ACT - G15.*RFP + G15.*T./(T+G7).*((A./Ce).^m./(1+(A./Ce).^m) + (B./Ce).^n./(1+(B./Ce).^n)).*GK + mCh_b;


% ------------------------------
%%Integrating three parts together
% ------------------------------
% advection, diffusion and reaction a,d,r stand for them respectively

dCedt  = dCedt_r; 
dNudt  = dNudt_r;            
dAdt   = dAdt_r;  
dBdt   = dBdt_r;

dLydt  = dLydt_r; 
dTdt   = dTdt_r;  
dPdt   = dPdt_r;  

dRFPdt = dRFPdt_r;
dCFPdt = dCFPdt_r;

% Assemble them into the output vector,send it back to the main function
out = [dCedt; dNudt; dAdt; dBdt; dLydt; dTdt; dPdt; dRFPdt; dCFPdt];

end


