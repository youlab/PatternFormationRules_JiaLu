
function out = get_C0_scale_v_od(param, opt, v, OD)

% Get initial C0 based on cell density and seeding volumn

L = param.L; 
N = param.N;

% Base case: V = 0.1 & OD = 0.2
R0 = 0.05;
radius = R0 * L; 
sigma = 0.3 * radius; 
xx = linspace(0, L, N);
out = 0.2 * exp(-((xx - radius).^2 / (2 * sigma^2)));
cell_num0 = trapz(xx, out);

if strcmp(opt, 'donut') 
     
    switch OD 
        case 0.02 

            r_scaling = [1.15, 1.74, 2.3, 2.57, 0,0,0,0,0,4.19];
            r_scale = r_scaling(round(v/0.1));
            
            % get r
            radius = r_scale * R0 * L; 
            sigma = 0.3 * radius; 
            xx = linspace(0, L, N);
            out = 0.2 * exp(-((xx - radius).^2 / (2 * sigma^2)));
            
            % get h
            cell_num =  cell_num0 * (OD /0.2) * r_scale^2; 
            area = trapz(xx, out);
            scaling_factor = cell_num/area;
            out = out * scaling_factor;
        
        case 0.2
            % get scaling factor
            r_scaling = [1, 1.3, 1.68, 1.81, 0,0,0,0,0,2.13];
            r_scale = r_scaling(round(v/0.1));

            % get r
            radius = r_scale * R0 * L; 
            sigma = 0.3 * radius; 
            xx = linspace(0, L, N);
            out = 0.2 * exp(-((xx - radius).^2 / (2 * sigma^2)));
            
            % get h
            cell_num =  cell_num0 * (OD /0.2) * r_scale^2; 
            area = trapz(xx, out);
            scaling_factor = cell_num/area;
            out = out * scaling_factor;

        case 1
            % get scaling factor
            r_scaling = [0.85, 1.24, 1.55, 1.87, 0,0,0,0,0,2.15];
            r_scale = r_scaling(round(v/0.1));

            % get r
            radius = r_scale * R0 * L; 
            sigma = 0.3 * radius; 
            xx = linspace(0, L, N);
            out = 0.2 * exp(-((xx - radius).^2 / (2 * sigma^2)));
            
            % get h
            cell_num =  cell_num0 * (OD /0.2) * r_scale^2; 
            area = trapz(xx, out);
            scaling_factor = cell_num/area;
            out = out * scaling_factor;
   
end

end
end
