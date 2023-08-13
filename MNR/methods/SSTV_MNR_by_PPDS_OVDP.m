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

function result = SSTV_MNR_by_PPDS_OVDP( ...
    DATA, ...
    params, ...
    stepsizes)
%% 
if ~isfield(params, 'use_GPU')
    use_GPU = 0;
else
    use_GPU = params.use_GPU;
end


%% Preparing data and parameters
clean_HSI = DATA.clean_HSI;
V = DATA.noised_HSI;
psuedo_true_HSI = DATA.psuedo_true_HSI;
psuedo_true_sparse_noise = DATA.psuedo_true_sparse_noise;
psuedo_true_stripe_noise = DATA.psuedo_true_stripe_noise;

if use_GPU == 1
    clean_HSI = gpuArray(clean_HSI);
    V = gpuArray(V);
    psuedo_true_HSI = gpuArray(psuedo_true_HSI);
    psuedo_true_sparse_noise = gpuArray(psuedo_true_sparse_noise);
    psuedo_true_stripe_noise = gpuArray(psuedo_true_stripe_noise);
end

max_iteration = params.max_iteration;
stopping_criterion = params.stopping_criterion;
rate_of_sparse = params.rate_of_sparse;
sigma_of_gaussian = params.sigma_of_gaussian;
lambda_L = params.lambda_L;

%% Setting parameters
n1 = size(clean_HSI, 1);
n2 = size(clean_HSI, 2);
n3 = size(clean_HSI, 3);

rate_eta = 0.95;
eta = rate_eta*rate_of_sparse*numel(clean_HSI)/2;
rate_epsilon = 0.95;
epsilon = rate_epsilon*sigma_of_gaussian*sqrt(numel(clean_HSI)*(1 - rate_of_sparse));


%% Setting stepsizes
Gamma1_U  = stepsizes.Gamma1_U;
Gamma1_S  = stepsizes.Gamma1_S;
Gamma1_L  = stepsizes.Gamma1_L;
Gamma2_Y1 = stepsizes.Gamma2_Y1;
Gamma2_Y2 = stepsizes.Gamma2_Y2;
Gamma2_Y3 = stepsizes.Gamma2_Y3;
Gamma2_Y4 = stepsizes.Gamma2_Y4;

%% Optimization

if use_GPU == 1
    U = gpuArray(zeros(n1, n2, n3));
    S = gpuArray(zeros(n1, n2, n3));
    L = gpuArray(zeros(n1, n2, n3));
    Y1   = gpuArray(zeros(n1, n2, n3));
    Y2   = gpuArray(zeros(n1, n2, n3));
    Y3   = gpuArray(zeros(n1, n2, n3));
    Y4   = gpuArray(V);
else
    U = zeros(n1, n2, n3);
    S = zeros(n1, n2, n3);
    L = zeros(n1, n2, n3);
    Y1   = zeros(n1, n2, n3);
    Y2   = zeros(n1, n2, n3);
    Y3   = zeros(n1, n2, n3);
    Y4   = V;
end

is_converged = 0;
iteration = 0;
interval_disp = 100;

vals_func = nan(1, max_iteration);
vals_run_time = nan(1, max_iteration);
vals_PSNR = nan(1, max_iteration);
distances_to_GT = nan(1, max_iteration);

val_run_time = 0;

while is_converged == 0
            
    timer_a = tic;
    % Updating primal variables
    U_next = U - Gamma1_U.*(D3t(D1t(Y1)) + D3t(D2t(Y2)) + Y4);
    S_next = ProjFastL1Ball(S - Gamma1_S.*(Y4), eta);
    L_next = ProxL1(L - Gamma1_L.*(D1t(Y3) + Y4), lambda_L*Gamma1_L);
    
    % Updating dual variables
    Y1_tmp   = Y1   + Gamma2_Y1.*D1(D3(2*U_next - U));
    Y2_tmp   = Y2   + Gamma2_Y2.*D2(D3(2*U_next - U));
    Y3_tmp   = Y3   + Gamma2_Y3.*D1(2*L_next - L);
    Y4_tmp   = Y4   + Gamma2_Y4.*(2*(U_next + S_next + L_next) - (U + S + L));
    
    Y1_next   = Y1_tmp   - Gamma2_Y1.*ProxL1(Y1_tmp./Gamma2_Y1, 1./Gamma2_Y1);
    Y2_next   = Y2_tmp   - Gamma2_Y2.*ProxL1(Y2_tmp./Gamma2_Y2, 1./Gamma2_Y2);
    Y3_next   = Y3_tmp;
    Y4_next   = Y4_tmp   - Gamma2_Y4.*ProjL2Ball(Y4_tmp./Gamma2_Y4, V, epsilon);
    
    val_run_time = val_run_time + toc(timer_a);
    
    % Calculating the distance between the psuedo optimal values and
    % the current variables
    distance_to_GT = gather(sqrt((sum((U_next - psuedo_true_HSI).^2, "all") ...
        + sum((S_next - psuedo_true_sparse_noise).^2, "all") ...
        + sum((L_next - psuedo_true_stripe_noise).^2, "all"))/(3*n1*n2*n3)));        
    
    % Calculating the objective function value of the current variables
    val_func = sum(abs(D1(D3(U_next))), "all") ...
        + sum(abs(D2(D3(U_next))), "all") ...
        + lambda_L*sum(abs(L_next), "all");
    diff_GT_U = clean_HSI - U_next;
    psnr_bandwise = 20*log10(sqrt(n2*n1)./reshape(sqrt(sum(sum(diff_GT_U.*diff_GT_U, 1), 2)), [1, n3]));
    
    if (iteration >= max_iteration || distance_to_GT < stopping_criterion)
        is_converged = 1;
    end
    
    timer_a = tic;
    % Updating
    U  = U_next;
    S  = S_next;
    L  = L_next;
    
    Y1 = Y1_next;
    Y2 = Y2_next;
    Y3 = Y3_next;
    Y4 = Y4_next;
    val_run_time = val_run_time + toc(timer_a);

    % Saving optimization status
    iteration = iteration + 1;
    vals_func(iteration) = val_func;
    vals_PSNR(iteration) = gather(mean(psnr_bandwise));
    distances_to_GT(iteration) = distance_to_GT;
    vals_run_time(iteration) = gather(val_run_time);
    
    if (mod(iteration, interval_disp) == 0)
        disp(append(...
            'iteration : ', num2str(iteration), ', ', ...
            'error : ', num2str(distance_to_GT), ', ', ...
            'function value : ', num2str(val_func), ', ', ...
            'MPSNR : ', num2str(gather(mean(psnr_bandwise)))));
    end
end

restorated_HSI = gather(U);

%% 
result.restorated_HSI = restorated_HSI;
result.vals_PSNR = vals_PSNR;
result.vals_func = vals_func;
result.vals_run_time = vals_run_time;
result.distances_to_GT = distances_to_GT;


end
