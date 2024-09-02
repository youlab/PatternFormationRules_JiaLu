function new = perturbParam(param, param_name, perturbationScale, opt)

rng('shuffle'); rngState = rng;
seed = rngState.Seed + uint32(feature('getpid'));
rng(seed);

if strcmp(opt, "linear")

    while true
        new = param.(param_name) + 0.2*perturbationScale * (2*rand()-1);
        if new > 0
            param.(param_name) = new;
            break
        end
    end

elseif strcmp(opt, "exp")

    while true
        pertubation = 10.^(0.2 * rand())/10;
        new = param.(param_name) + perturbationScale * pertubation;
        if new > 0
            param.(param_name) = new;
            break
        end
    end
    
end