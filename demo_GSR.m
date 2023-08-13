warning off all
clear;
close all;
rmpath(genpath('./'));

addpath(genpath('./GSR'));

%% Loading data
load('./datasets/dataset_GSR.mat');

p_true_func_val = p_func_val;

stopping_criterion = 1e-10;
max_iteration = 20000;

use_GPU = 0; % if you use GPUs, set use_GPU = 1

%% Calculating stepsizes
beta = 1;
WD = G.Diff;
sing_val_WD = svds(WD, 1);
if beta == 0
    stepsizes.Gamma1_u  = 1/(sing_val_WD.^2 + 1);
    stepsizes.Gamma2_y1 = 1;
    stepsizes.Gamma2_y2 = 1;
elseif beta == 2
    stepsizes.Gamma1_u  = 1/2;
    stepsizes.Gamma2_y1 = 1/(sing_val_WD.^2);
    stepsizes.Gamma2_y2 = 1;
elseif beta > 0 && beta < 2
    stepsizes.Gamma1_u  = 1/(sing_val_WD^(2 - beta) + 1^(2 - beta));
    stepsizes.Gamma2_y1 = 1/(sing_val_WD^beta);
    stepsizes.Gamma2_y2 = 1/(1^beta);
else
    disp('The beta is invalid.');
end

%% setting Images and parameters
DATA.x_noisy = x_noisy;
DATA.x = x;
DATA.G = G;
DATA.mask = mask;
DATA.psuedo_true_signal = psuedo_true_signal;
DATA.Group_index_matrix = Group_index_matrix;
DATA.Group_index_matrix_func = Group_index_matrix_func;

params.max_iteration = max_iteration;
params.stopping_criterion = stopping_criterion;
params.use_GPU = use_GPU;
params.noise_sigma = noise_sigma;
params.p_p = p_p;

%% DP-PDS with new stepsize adjustment
results = GSR_by_PPDS_OVDP(DATA, params, stepsizes);

distances_to_GT = results.distances_to_GT;
vals_func = results.vals_func;
vals_run_time = results.vals_run_time;
vals_PSNR = results.vals_PSNR;
u_est = results.u_est;

%% plot
fig = figure;
fig.Position(2) = 100;
fig.Position(3) = 2200;
fig.Position(4) = 800;

max_x_axis = max_iteration;

step_plot = 100;
x_lim_time = 10;

size_font = 20;
size_font_title = 25;

width_line = 2;


%% distance vs iteration
subplot(2, 3, 1)

loglog(...
    1:step_plot:max_iteration, distances_to_GT(1, 1:step_plot:max_iteration), '-' , ...
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
    1:step_plot:max_iteration, abs(p_true_func_val - vals_func(1, 1:step_plot:max_iteration)) , ...
    'LineWidth', width_line);

ylabel("RES", 'FontSize', size_font, 'FontWeight', 'bold');
xlabel("Iterations k", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("Iteration vs RES", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([2 max_x_axis])

%% PSNR vs iteration
subplot(2, 3, 3)

semilogx(...
    1:step_plot:max_iteration, vals_PSNR(1, 1:step_plot:max_iteration), ...
    'LineWidth', width_line);

ylabel("PSNR", 'FontSize', size_font, 'FontWeight', 'bold');
xlabel("Iterations k", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("Iteration vs PSNR", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([2 max_x_axis])

%% Distance vs times
subplot(2, 3, 4)

semilogy(...
    vals_run_time(1, 1:step_plot:max_iteration), distances_to_GT(1, 1:step_plot:max_iteration), ...
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
    vals_run_time(1, 1:step_plot:max_iteration), abs(p_true_func_val - vals_func(1, 1:step_plot:max_iteration)), ...
    'LineWidth', width_line);

ylabel("RES", 'FontSize', size_font, 'FontWeight', 'bold');
xlabel("Time [s]", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("Computational time vs RES", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([0 x_lim_time])

%% PSNR vs time
subplot(2, 3, 6)

plot(...
    vals_run_time(1, 1:step_plot:max_iteration), vals_PSNR(1, 1:step_plot:max_iteration), ...
    'LineWidth', width_line);

ylabel("PSNR", 'FontSize', size_font, 'FontWeight', 'bold');
xlabel("Time [s]", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("Computational time vs PSNR", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([0 x_lim_time])

%% Plotting the recovered graph signal
figure;
param.N = G.N;
param.colorbar = 0;
gsp_plot_signal(G, u_est, param);
colorbar('northoutside');
clim([0 1]);
set(gca, 'FontSize', size_font);
