function out = get_C0_scale(param, opt)

% assume L is fixed, N is changed to adjust grid size

L = param.L; 
N = param.N;

if strcmp(opt, 'donut')
    radius = 0.01 * L;
    sigma = 0.3 * radius; % Sigma for Gaussian distribution
    
    xx = linspace(0, L, N);
    out = 0.2 * exp(-((xx - radius).^2 / (2 * sigma^2)));

elseif strcmp(opt, 'dome')
    xx = linspace(0, L, N);
    out = 0.2 * exp(-(xx).^2 / (1/2));
   
end

end
