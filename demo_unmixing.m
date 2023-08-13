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

warning off all
clear;
close all;
rmpath(genpath('./'))

addpath(genpath('./unmixing'));

%% Loading data
load('./datasets/dataset_unmixing.mat');

stopping_criterion = 1e-10;
max_iteration = 20000;

use_GPU = 0; % if you use GPUs, set use_GPU = 1

%% Calculating stepsizes
beta = 1;
sing_val_em = max(svd(Endmember));
if beta == 0
    stepsizes.Gamma1_A  = 1/(sing_val_em.^2 + 1);
    stepsizes.Gamma2_Y1 = 1;
    stepsizes.Gamma2_Y2 = 1;
elseif beta == 2
    stepsizes.Gamma1_A  = 1/2;
    stepsizes.Gamma2_y1 = 1/(sing_val_em.^2);
    stepsizes.Gamma2_Y2 = 1;
elseif beta > 0 && beta < 2
    stepsizes.Gamma1_A  = 1/(sing_val_em^(2 - beta) + 1^(2 - beta));
    stepsizes.Gamma2_Y1 = 1/(sing_val_em^beta);
    stepsizes.Gamma2_Y2 = 1/(1^beta);
else
    disp('The beta is invalid.');
end

%% setting Images and parameters
DATA.HSI_GT = HSI_GT;
DATA.HSI_NOISY = HSI_NOISY;
DATA.A_p_true = A_p_true;
DATA.Abandance = Abandance;
DATA.Endmember = Endmember;

params.sigma_Gaussian = sigma_Gaussian;
params.num_endmember = num_endmember;
params.max_iteration = max_iteration;
params.stopping_criterion = stopping_criterion;
params.use_GPU = use_GPU;

%% DP-PDS with new stepsize adjustment
results = unmixing_by_PPDS_OVDP(DATA, params, stepsizes);

distances_to_GT = results.distances_to_GT;
vals_func = results.vals_func;
vals_run_times = results.vals_run_times;
vals_SRE = results.vals_SRE;
ABUNDANCE_EST = results.ABUNDANCE_EST;

%% Plotting
fig = figure;
fig.Position(2) = 100;
fig.Position(3) = 2200;
fig.Position(4) = 1200;

max_x_axis = max_iteration;

step_plot = 100;
x_lim_time = 15;

size_font = 20;
size_font_title = 25;

width_line = 2;

%% Distance vs iteration
subplot(2, 3, 1)

loglog(...
    1:step_plot:max_iteration, distances_to_GT(1, 1:step_plot:max_iteration), ...
    'LineWidth', width_line);

ylabel("RMSE", 'FontSize', size_font, 'FontWeight', 'bold');
xlabel("Iterations k", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("Iteration vs RMSE", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([2 max_x_axis])

%% RES vs iteration
subplot(2, 3, 2)

loglog(...
    1:step_plot:max_iteration, abs(p_true_func_val - vals_func(1, 1:step_plot:max_iteration)), ...
    'LineWidth', width_line);

ylabel("Residual", 'FontSize', size_font, 'FontWeight',' bold');
xlabel("Iterations k", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("Iteration vs Residual", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([2 max_x_axis])

%% SRE vs iteration
subplot(2, 3, 3)

semilogx(...
    1:step_plot:max_iteration, vals_SRE(1, 1:step_plot:max_iteration), ...
    'LineWidth', width_line);

ylabel("SNR", 'FontSize', size_font, 'FontWeight', 'bold');
xlabel("Iterations k", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("Iteration vs SNR", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([2 max_x_axis])

%% Distance vs times
subplot(2, 3, 4)

semilogy(...
    vals_run_times(1, 1:step_plot:max_iteration), distances_to_GT(1, 1:step_plot:max_iteration), ...
    'LineWidth', width_line);

ylabel("RMSE", 'FontSize', size_font, 'FontWeight', 'bold');
xlabel("Time [s]", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("Computational time vs RMSE", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([0 x_lim_time])

%% RES vs time
subplot(2, 3, 5)

semilogy(...
    vals_run_times(1, 1:step_plot:max_iteration), abs(p_true_func_val - vals_func(1, 1:step_plot:max_iteration)), ...
    'LineWidth', width_line);

ylabel("Residual", 'FontSize', size_font, 'FontWeight', 'bold');
xlabel("Time [s]", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("Computational time vs Residual", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([0 x_lim_time])

%% SRE vs time
subplot(2, 3, 6)

fig = plot(...
    vals_run_times(1, 1:step_plot:max_iteration), vals_SRE(1, 1:step_plot:max_iteration), ...
    'LineWidth', width_line);

ylabel("SNR", 'FontSize', size_font, 'FontWeight', 'bold');
xlabel("Time [s]", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("(c2) Computational time vs SNR", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);
xlim([0 x_lim_time])

%% Plotting the estimated abundance maps
fig_abun = figure;
fig_abun.Position(2) = 100;
fig_abun.Position(3) = 1200;
fig_abun.Position(4) = 800;

for idx_abun = 1:6
    subplot(2, 3, idx_abun);
    imagesc(ABUNDANCE_EST(:, :, idx_abun));
    title(cood{idx_abun}, 'FontSize', 20);
end
sgtitle('Unmixing results', 'FontSize', 20);


