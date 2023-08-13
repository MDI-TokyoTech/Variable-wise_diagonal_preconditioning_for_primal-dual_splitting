function results = unmixing_by_PPDS_OVDP( ...
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
HSI_GT = DATA.HSI_GT;
V = DATA.HSI_NOISY;
A_p_true = DATA.A_p_true;
Abandance = DATA.Abandance;
Endmember = DATA.Endmember;

if use_GPU == 1
    HSI_GT = gpuArray(HSI_GT);
    V = gpuArray(V);
    A_p_true = gpuArray(A_p_true);
    Abandance = gpuArray(Abandance);
    Endmember = gpuArray(Endmember);
end

sigma_Gaussian = params.sigma_Gaussian;
num_endmember = params.num_endmember;
max_iteration = params.max_iteration;
stopping_criterion = params.stopping_criterion;

%% Setting parameters
n1 = size(HSI_GT, 1);
n2 = size(HSI_GT, 2);
n3 = size(HSI_GT, 3);

rate_epsilon = 0.9;
epsilon = rate_epsilon*sigma_Gaussian*sqrt(n1*n2*n3);

%% Setting stepsizes
Gamma1_A  = stepsizes.Gamma1_A;
Gamma2_Y1 = stepsizes.Gamma2_Y1;
Gamma2_Y2 = stepsizes.Gamma2_Y2;

%% Optimization

V = transpose(reshape(V, [n1*n2, n3]));
E = Endmember;


if use_GPU == 1
    A = gpuArray(ones(num_endmember, n1*n2));
    Y1   = gpuArray(zeros(n3, n1*n2));
    Y2   = gpuArray(zeros(num_endmember, n1*n2));
else
    A = ones(num_endmember, n1*n2);
    Y1   = zeros(n3, n1*n2);
    Y2   = zeros(num_endmember, n1*n2);
end

is_converged = 0;
iteration = 0;
interval_disp = 100;

vals_func = zeros(1, max_iteration);
vals_run_times = zeros(1, max_iteration);
vals_SRE = zeros(1, max_iteration);
distances_to_GT = zeros(1, max_iteration);

val_run_time = 0;

while is_converged == 0
    
    tic
    % Updating primal variables
    A_next = ProxL21norm(A - Gamma1_A.*(E'*Y1 + Y2), Gamma1_A);
    
    % Updating dual variables
    Y1_tmp   = Y1   + Gamma2_Y1.*E*(2*A_next - A);
    Y2_tmp   = Y2   + Gamma2_Y2.*(2*A_next - A);
    
    Y1_next   = Y1_tmp   - Gamma2_Y1.*ProjL2Ball(Y1_tmp./Gamma2_Y1, V, epsilon);
    Y2_next   = Y2_tmp   - Gamma2_Y2.*ProjNNR(Y2_tmp./Gamma2_Y2);
    
    val_run_time = val_run_time + toc;    
    
    % Calculating the distance between the psuedo optimal values and
    % the current variables
    distance_to_GT = gather(sqrt(sum((A_next - A_p_true).^2, "all")/(n1*n2*num_endmember)));

    % Calculating the objective function value of the current variables
    val_func = gather(L12norm(A));
    
    if (distance_to_GT < stopping_criterion || max_iteration < iteration)
        is_converged = 1;
    end
    
    tic 
    % Updating
    A  = A_next;
    
    Y1 = Y1_next;
    Y2 = Y2_next;
    val_run_time = val_run_time + toc;

    % Saving optimization status
    iteration = iteration + 1;
    vals_func(iteration) = val_func;
    vals_SRE(iteration) = sre(A, Abandance);
    distances_to_GT(iteration) = distance_to_GT;
    vals_run_times(iteration) = val_run_time;
    
    if (mod(iteration, interval_disp) == 0)
        disp(append(...
            "iteration : ", num2str(iteration), ', ', ...
            "error : ", num2str(distance_to_GT), ", ", ...
            "function value : ", num2str(val_func), ', ', ...
            "SRE : ", num2str(sre(A, Abandance))));
    end
end

ABUNDANCE_EST = gather(reshape(A', n1, n2, num_endmember));

%%
results.distances_to_GT = distances_to_GT;
results.vals_run_times = vals_run_times;
results.vals_SRE = vals_SRE;
results.vals_func = vals_func;
results.ABUNDANCE_EST = ABUNDANCE_EST;

end
