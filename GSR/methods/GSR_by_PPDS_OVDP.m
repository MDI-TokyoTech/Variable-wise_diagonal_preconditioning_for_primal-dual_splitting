%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the demo file of the method proposed in the
% following reference:
% 
% K. Naganuma and S. Ono
% ``Variable-Wise Diagonal Preconditioning for Primal-Dual Splitting: Design and Applications''
%
% Update history:
% Augast 14, 2023: v1.0 
%
% Copyright (c) 2023 Kazuki Naganuma and Shunsuke Ono
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function results = GSR_by_PPDS_OVDP(...
    DATA, ...
    params, ...
    stepsizes)

%% 
G = DATA.G;

if ~isfield(params, 'use_GPU')
    use_GPU = 0;
else
    use_GPU = params.use_GPU;
end

%% Preparing data and parameters
x_noisy = DATA.x_noisy;
u_org = DATA.x;
mask = DATA.mask;
WD = G.Diff;

psuedo_true_signal = DATA.psuedo_true_signal;
Group_index_matrix = DATA.Group_index_matrix;
Group_index_matrix_func = DATA.Group_index_matrix_func;

max_iteration = params.max_iteration;
stopping_criterion = params.stopping_criterion;
noise_sigma = params.noise_sigma;
p_p = params.p_p;

if use_GPU == 1
    x_noisy = gpuArray(x_noisy);
    u_org = gpuArray(u_org);
    mask = gpuArray(mask);
    psuedo_true_signal = gpuArray(psuedo_true_signal);
    Group_index_matrix = gpuArray(Group_index_matrix);
    Group_index_matrix_func = gpuArray(Group_index_matrix_func);
    WD = gpuArray(full(WD));
end

v = mask.*x_noisy;

phi = @(z) mask.*z;
phiT = phi;

num_sig = numel(u_org);
rate_epsilon = 0.9;
epsilon = rate_epsilon*noise_sigma*p_p*sqrt(num_sig);

%% Setting stepsizes
Gamma1_u  = stepsizes.Gamma1_u;
Gamma2_y1 = stepsizes.Gamma2_y1;
Gamma2_y2 = stepsizes.Gamma2_y2;

%% Optimization
if use_GPU == 1
    u = gpuArray(zeros(size(v)));
else
    u = zeros(size(v));
end

y1   = WD*u;
y2   = phi(u);

is_converged = 0;
iteration = 0;
interval_disp = 100;

vals_func = zeros(1, max_iteration);
vals_run_time = zeros(1, max_iteration);
vals_PSNR = zeros(1, max_iteration);
distances_to_GT = zeros(1, max_iteration);

val_run_time = 0;

while is_converged == 0
    
    tic
    % Updating primal variables
    u_next = u - Gamma1_u.*(WD'*y1 + phiT(y2));
    
    % Updating dual variables
    y1_tmp   = y1   + Gamma2_y1.*WD*(2*u_next - u);
    y2_tmp   = y2   + Gamma2_y2.*phi(2*u_next - u);
    
    y1_next   = y1_tmp   - Gamma2_y1.*ProxMixedL12NormGraph(y1_tmp./Gamma2_y1, 1./Gamma2_y1, Group_index_matrix);
    y2_next   = y2_tmp   - Gamma2_y2.*ProjL2Ball(y2_tmp./Gamma2_y2, v, epsilon);
    
    val_run_time = val_run_time + toc;
    
    % Calculating the distance between the psuedo optimal values and
    % the current variables
    distance_to_GT = gather(sqrt(sum((u_next - psuedo_true_signal).^2, "all")/(num_sig)));
    
    % Calculating the objective function value of the current variables
    val_func = gather(MixedL12NormGraph(WD*u, Group_index_matrix_func));
    
    if (max_iteration < iteration || distance_to_GT < stopping_criterion)
        is_converged = 1;
    end

    tic 
    % Updating
    u  = u_next;
    
    y1 = y1_next;
    y2 = y2_next;
    
    val_run_time = val_run_time + toc;

    % Saving optimization status
    iteration = iteration + 1;
    vals_func(iteration) = val_func;
    vals_PSNR(iteration) = gather(psnr(u, u_org));
    distances_to_GT(iteration) = distance_to_GT;
    vals_run_time(iteration) = gather(val_run_time);
    
    if (mod(iteration, interval_disp) == 0)
        disp(append(...
            'iteration : ', num2str(iteration), ', ', ...
            'error : ', num2str(distance_to_GT), ', ', ...
            'function value : ', num2str(val_func), ', ', ...
            'PSNR : ', num2str(psnr(u, u_org))));
    end
end

u_est = gather(u);

results.u_est = u_est;
results.vals_PSNR = vals_PSNR;
results.vals_func = vals_func;
results.vals_run_time = vals_run_time;
results.distances_to_GT = distances_to_GT;

end

