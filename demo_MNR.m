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
rmpath(genpath('./'));

addpath(genpath('./MNR'));

%% Loading data
load("./datasets/dataset_MNR.mat");

p_true_func_val = psuedo_true_function_value;

stopping_criterion = 1e-10;
max_iteration = 20000;

use_GPU = 0; % if you use GPUs, set use_GPU = 1

%% Calculating stepsizes
beta = 1;
if beta == 0
    stepsizes.Gamma1_U  = 1/(4^2 + 4^2 + 1);
    stepsizes.Gamma1_S  = 1;
    stepsizes.Gamma1_L  = 1/(2^2 + 1);
    stepsizes.Gamma2_Y1 = 1/3;
    stepsizes.Gamma2_Y2 = 1/3;
    stepsizes.Gamma2_Y3 = 1/3;
    stepsizes.Gamma2_Y4 = 1/3;
elseif beta == 2
    stepsizes.Gamma1_U  = 1/4;
    stepsizes.Gamma1_S  = 1/4;
    stepsizes.Gamma1_L  = 1/4;
    stepsizes.Gamma2_Y1 = 1/(4^2);
    stepsizes.Gamma2_Y2 = 1/(4^2);
    stepsizes.Gamma2_Y3 = 1/(2^2);
    stepsizes.Gamma2_Y4 = 1/(1^2 + 1^2 + 1^2);
elseif beta > 0 && beta < 2
    stepsizes.Gamma1_U  = 1/(4^(2 - beta) + 4^(2 - beta) + 1^(2 - beta));
    stepsizes.Gamma1_S  = 1/(1^(2 - beta));
    stepsizes.Gamma1_L  = 1/(2^(2 - beta) + 1^(2 - beta));
    stepsizes.Gamma2_Y1 = 1/(4^beta);
    stepsizes.Gamma2_Y2 = 1/(4^beta);
    stepsizes.Gamma2_Y3 = 1/(2^beta);
    stepsizes.Gamma2_Y4 = 1/(1^beta + 1^beta + 1^beta);
else
    disp('The beta is invalid.')
end

%% Setting Images and parameters
DATA.clean_HSI = clean_HSI;
DATA.noised_HSI = noised_HSI;
DATA.psuedo_true_HSI = psuedo_true_HSI;
DATA.psuedo_true_sparse_noise = psuedo_true_sparse_noise;
DATA.psuedo_true_stripe_noise = psuedo_true_stripe_noise;

params.max_iteration = max_iteration;
params.stopping_criterion = stopping_criterion;
params.rate_of_sparse = rate_of_sparse;
params.sigma_of_gaussian = sigma_of_gaussian;
params.lambda_L = lambda_L;
params.use_GPU = use_GPU;


%% Mixed noise removal
results = SSTV_MNR_by_PPDS_OVDP(DATA, params, stepsizes);

distances_to_GT = results.distances_to_GT;
vals_func = results.vals_func;
vals_run_times = results.vals_run_time;
vals_PSNR = results.vals_PSNR;
restorated_HSI = results.restorated_HSI;

%% Ploting
fig = figure;
fig.Position(2) = 100;
fig.Position(3) = 2200;
fig.Position(4) = 800;

max_x_axis = max_iteration;
range_y_RMSE = [0.001, 1];
range_y_RES = [1e-2, 1230780];
range_y_PSNR = [0, 36];

step_plot = 100;
x_lim_time = 120;

size_font = 20;
size_font_title = 25;

width_line = 2;


%% Distance vs iteration
subplot(2, 3, 1)

loglog(...
    1:step_plot:max_iteration, distances_to_GT(1, 1:step_plot:max_iteration), ...
    'LineWidth', width_line);

ylabel("RMSE", 'FontSize', size_font, 'FontWeight','bold');
xlabel("Iterations k", 'FontSize', size_font, 'FontWeight', 'bold');
set(gca, 'FontSize', size_font);
title("Iteration vs RMSE", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([2 max_x_axis])
ylim(range_y_RMSE);

%% RES vs iteration
subplot(2, 3, 2)

loglog(...
    1:step_plot:max_iteration, abs(p_true_func_val - vals_func(1, 1:step_plot:max_iteration)), ...
    'LineWidth', width_line);

ylabel("RES", 'FontSize', size_font, 'FontWeight','bold');
xlabel("Iterations k", 'FontSize',size_font,'FontWeight','bold');
set(gca, 'FontSize', size_font);
title("Iteration vs RES", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([2 max_x_axis])
ylim(range_y_RES);

%% PSNR vs iteration
subplot(2, 3, 3)

semilogx(...
    1:step_plot:max_iteration, vals_PSNR(1, 1:step_plot:max_iteration), ...
    'LineWidth', width_line);


ylabel("PSNR", 'FontSize', size_font, 'FontWeight', 'bold');
xlabel("Iterations", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("Iteration vs PSNR", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([2 max_x_axis])
ylim(range_y_PSNR);

%% Distance vs times
subplot(2, 3, 4)

semilogy(...
    vals_run_times(1, 1:step_plot:max_iteration), distances_to_GT(1, 1:step_plot:max_iteration), ...
    'LineWidth', width_line);

ylabel("RMSE", 'FontSize', size_font, 'FontWeight','bold');
xlabel("Time [s]", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("Computational time vs RMSE", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([0 x_lim_time])
ylim(range_y_RMSE);

%% RES vs time
subplot(2, 3, 5)

semilogy(...
    vals_run_times(1, 1:step_plot:max_iteration), abs(p_true_func_val - vals_func(1, 1:step_plot:max_iteration)), ...
    'LineWidth', width_line);


ylabel("RES", 'FontSize', size_font, 'FontWeight', 'bold');
xlabel("Time [s]", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("Computational time vs RES", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([0 x_lim_time])
ylim(range_y_RES);

%% PSNR vs time
subplot(2, 3, 6)

fig = plot(...
    vals_run_times(1, 1:step_plot:max_iteration), vals_PSNR(1, 1:step_plot:max_iteration), ...
    'LineWidth', width_line);

ylabel("PSNR", 'FontSize', size_font, 'FontWeight', 'bold');
xlabel("Time [s]", 'FontSize', size_font, 'FontWeight', 'bold');

set(gca, 'FontSize', size_font);
title("Computational time vs PSNR", ...
    'FontName', 'Times New Roman', ...
    "FontSize", size_font_title);

xlim([0 x_lim_time])
ylim(range_y_PSNR);

%% Plotting the restorated image
figure;
imshow(restorated_HSI(:, :, 100));
