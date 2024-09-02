function output = get_rand(min, max, num, opt)

% num is the length of screening list

rng('shuffle'); rngState = rng;
seed = rngState.Seed + uint32(feature('getpid')); 
rng(seed);

if strcmp(opt, 'linear')
    output = (max-min).*rand(1, num) + min;

elseif strcmp(opt, 'log')

    log_min = log10(min); 
    log_max = log10(max);
    output = 10.^(log_min + (log_max-log_min) * rand(1,num));   

elseif strcmp(opt, 'int')
    output = randi([min max], 1, num);

elseif strcmp(opt, 'n')

    n_idx = randi([min max], 1, num);
    n_list = [0.1, 0.25, 0.5, 1, 2.5, 5];
    output = n_list(n_idx);
    
end

end

