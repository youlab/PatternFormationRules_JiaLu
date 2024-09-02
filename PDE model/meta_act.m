% This is the metabolic capacity function

function out = meta_act(Ce, Nu, Ly, T, param)

alpha_p = param.alpha_p;
beta_p = param.beta_p;
G2 = param.G2;

out = (1-Ce).*Nu./(G2+Nu)./(1+alpha_p*T+beta_p*Ly);
