% for generating the comprehensive dataset

% close all
clear all

taskID = str2num(getenv('SLURM_ARRAY_TASK_ID'))

% Directory for data storage
data_dir = pwd;
if exist(data_dir, 'dir') ~= 7
    mkdir(data_dir);
end

%% parameters
parameters_augment;

Ce_pre = param.Ce0;
Nu_pre = param.Nu0;
A_pre  = param.A0;
B_pre  = param.B0;
Ly_pre = param.Ly0;
T_pre  = param.T0;
P_pre  = param.P0;
RFP_pre = param.RFP0;
CFP_pre = param.CFP0;

%% Solve PDE
nt = round(param.tmax / param.dt);
t  = 0;

relTolValue = 1e-4;  % Relative tolerance
absTolValue = 1e-6;  % Absolute tolerance
options = odeset('RelTol', relTolValue, 'AbsTol', absTolValue, 'InitialStep',0.01);

N = param.N;
L = param.L;
h = param.L / (param.N-1);

%% --------------------- Implicit method ---------------------

% Diffusion

DOMC = diffusion1Dx(param.G1, N, h, param.dt);
DOMN = diffusion1Dx(param.G3, N, h, param.dt);
DOMA = diffusion1Dx(param.G5, N, h, param.dt);
DOMB = diffusion1Dx(param.G9, N, h, param.dt);

AOM = getaom(N, param.L);

% Advection
beta1 = param.G1 * param.dt / h;


tic

for i = 1:nt % time step
    t = t + param.dt
    % ------------------------------------------------------------
    % Solve PDE
    % ------------------------------------------------------------
    % Solve advection -> diffusion -> reaction

    % -------------------------Advection-----------------------------------
    fprintf('advection')

    [Ly_pre,T_pre,P_pre,RFP_pre,CFP_pre] = Advect(Ly_pre,T_pre,P_pre,RFP_pre,CFP_pre,Ce_pre,AOM,param,param.dt);

    % Ensure non-negative values
    Ly_pre = max(Ly_pre, 0);
    T_pre = max(T_pre, 0);
    P_pre = max(P_pre, 0);
    RFP_pre = max(RFP_pre, 0);
    CFP_pre = max(CFP_pre, 0);

    % -------------------------Diffusion-----------------------------------
    fprintf('diffusion') %
    Ce_pre = DOMC \ Ce_pre;
    Nu_pre = DOMN \ Nu_pre;
    A_pre  = DOMA \ A_pre;
    B_pre  = DOMB \ B_pre;

    % Ensure non-negative values
    Ce_pre = max(Ce_pre, 0);
    Nu_pre = max(Nu_pre, 0);
    A_pre = max(A_pre, 0);
    B_pre = max(B_pre, 0);


    % -------------------------reaction-----------------------------------
    fprintf('reaction')

    inputs = [Ce_pre;Nu_pre;A_pre;B_pre;Ly_pre;T_pre;P_pre;RFP_pre;CFP_pre];
    sol = ode23(@adr_func_react_s, [0 param.dt], inputs, options, param, t);
    vec = (deval(sol, param.dt));

    % update initial values for the next interation
    Ce_pre = vec(1:N);
    Nu_pre = vec(N+1:2*N);
    A_pre  = vec(2*N+1:3*N);
    B_pre  = vec(3*N+1:4*N);
    Ly_pre = vec(4*N+1:5*N);
    T_pre  = vec(5*N+1:6*N);
    P_pre  = vec(6*N+1:7*N);
    RFP_pre = vec(7*N+1:8*N);
    CFP_pre = vec(8*N+1:9*N);

    Ce_pre = max(Ce_pre, 0);
    Nu_pre = max(Nu_pre, 0);
    A_pre = max(A_pre, 0);
    B_pre = max(B_pre, 0);
    Ly_pre = max(Ly_pre, 0);
    T_pre = max(T_pre, 0);
    P_pre = max(P_pre, 0);
    RFP_pre = max(RFP_pre, 0);
    CFP_pre = max(CFP_pre, 0);

    % ------------------------------------------------------------
    % Update accummulative fluorescent expression pattern
    % ------------------------------------------------------------
    % calculate total RFP distribution
    total_RFP = Ce_pre.*RFP_pre;
    total_CFP = Ce_pre.*CFP_pre;

    % ------------------------------------------------------------
    % Load the final results into the history vector for tracking
    % ------------------------------------------------------------
    % size = N, t
    hist_Ce(:, i) = Ce_pre;
    hist_Nu(:, i) = Nu_pre;
    hist_A(:,i)  = A_pre;
    hist_B(:,i)  = B_pre;
    hist_Ly(:,i) = Ly_pre;
    hist_T(:, i)  = T_pre;
    hist_P(:, i)  = P_pre;
    hist_aveRFP(:, i) = RFP_pre;
    hist_aveCFP(:, i) = CFP_pre;
    hist_RFP(:, i) = total_RFP;
    hist_CFP(:, i) = total_CFP;
    hist_t(:, i)   = t;

    
    if Ce_pre(end) >= 0.05*max(Ce_pre); %min(0.05*max(Ce_pre), 0.05)
        break
    end

end

toc

% Save plot data in param
param.hist_Ce = hist_Ce;
param.hist_Nu = hist_Nu;
param.hist_A = hist_A;
param.hist_B = hist_B;
param.hist_Ly = hist_Ly;
param.hist_T = hist_T;
param.hist_P = hist_P;
param.hist_aveRFP = hist_aveRFP;
param.hist_aveCFP = hist_aveCFP;
param.hist_RFP = hist_RFP;
param.hist_CFP = hist_CFP;
param.hist_t = hist_t;

%%  save multiple ring parameter set
filename = [data_dir, num2str(taskID) '.mat']
save(filename, "param");
