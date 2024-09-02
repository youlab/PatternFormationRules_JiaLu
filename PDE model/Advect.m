function [L,T,P,RFP,CFP] = Advect(L,T,P,RFP,CFP,C,AOM,param,dt)


N  = param.N;
G1 = param.G1;

% 1e-5 is there to avoid troubles with zero-values of Ce
CM = 1./(C + 1e-5) .* (AOM * C); 

AOM = speye(N) - G1 * dt * bsxfun(@times, CM, AOM);

L   = AOM \ L;
T   = AOM \ T;
P   = AOM \ P;
RFP = AOM \ RFP;
CFP = AOM \ CFP;






