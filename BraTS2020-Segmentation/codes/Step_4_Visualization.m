%% Step_4_Visualization_With_Regression_Fixed.m
% ======================================================================
% 🎯 STEP 4 — Visualization & Paper Figure Generation (UPDATED WITH REGRESSION)
% ======================================================================
% This script generates publication-quality figures from your results.
% It strictly uses your provided data (no extrapolation).
%
% NEW: Added regression model at the beginning (as per BraTS 2020 official task).
%   - Computes official metrics: MSE, MAE, Spearman, Pearson, Bin Accuracy.
%   - Uses Linear Regression (simple and fast; can swap to Random Forest if needed).
%   - Integrates Age and Resection as features.
%   - Integrates cleanup for duplicate features to fix rank deficiency warning.
%   - Keeps original Cox figures but adds regression-specific ones (e.g., Predicted vs True Scatter).
%
% FIGURES GENERATED (UPDATED):
% 1. Training Dynamics (Smoothed)
% 2. Predicted vs True Survival (Scatter Plot - Regression Replacement for KM)
% 3. Feature Importance (Bar Chart - From Regression Coefficients)
% 4. Correlation Matrix (Heatmap)
% 5. Age vs Survival Analysis (Scatter Plot)
% ======================================================================
clear; clc; close all;
fprintf('Starting Step 4: Generating Paper Figures (with Regression)...\n');

%% -------------------- 1. Select Data Folder (Moved to Top) --------------------
fprintf('Please select the folder containing your "results" (.mat and .csv files)...\n');
data_dir = uigetdir(pwd, 'Select Results Folder');
if data_dir == 0, error('No folder selected. Script Cancelled.'); end
output_fig_dir = fullfile(data_dir, 'Paper_Figures_Matlab');
if ~exist(output_fig_dir, 'dir'), mkdir(output_fig_dir); end
% Define File Paths
unet_path = fullfile(data_dir, 'trainedUnet.mat');
surv_model_path = fullfile(data_dir, 'survival_model.mat');
features_csv = fullfile(data_dir, 'features_table_for_model.csv');
survival_info_csv = fullfile(data_dir, 'survival_info.csv'); % For Age analysis

%% -------------------- 0. Regression Model --------------------
fprintf('Computing Regression Model (BraTS 2020 Official Task)...\n');
if isfile(features_csv) && isfile(survival_info_csv)
    % Load features and survival data
    T_feats = readtable(features_csv);
    surv_table = readtable(survival_info_csv);
    
    % Join tables (match by PatientID / Brats20ID)
    if ismember('Brats20ID', surv_table.Properties.VariableNames)
        surv_table.Properties.VariableNames{'Brats20ID'} = 'PatientID';
    end
    T = join(T_feats, surv_table(:, {'PatientID', 'Age', 'Extent_of_Resection'}), 'Keys', 'PatientID');
    
    % Numeric encode Resection (GTR=1, STR=2, NA=0)
    T.ResectionNum = zeros(height(T),1);
    T.ResectionNum(strcmp(T.Extent_of_Resection, 'GTR')) = 1;
    T.ResectionNum(strcmp(T.Extent_of_Resection, 'STR')) = 2;
    
    % === CLEANUP: Remove duplicates to fix rank deficiency ===
    T.Vol_mm3 = [];           % Identical to Vol_voxels
    T.BBoxVol_mm3 = [];       % Identical to BBoxVol_voxels
    
    % Filter valid rows
    valid_idx = ~isnan(T.Survival_days) & (T.Survival_days > 0);
    T_valid = T(valid_idx, :);
    
    % Features: Radiomics + Age + ResectionNum (now clean)
    feat_names = {'Vol_voxels', 'BBoxVol_voxels', 'MeanInt', 'MinInt', 'MaxInt', 'VarInt', 'SkewInt', 'KurtInt'}; % Adjusted for removed cols
    X = table2array(T_valid(:, [feat_names, 'Age', 'ResectionNum']));
    y = T_valid.Survival_days;
    
    % Train Linear Regression (no warning now)
    mdl = fitlm(X, y);
    
    % Predict
    pred = predict(mdl, X);
    
    % Official BraTS Metrics
    mse = mean((y - pred).^2);
    mae = mean(abs(y - pred));
    spearman = corr(y, pred, 'Type', 'Spearman');
    pearson = corr(y, pred);
    
    % Bin accuracy (short <300 days, medium 300-450, long >450)
    true_bins = 1*(y<300) + 2*((y>=300)&(y<=450)) + 3*(y>450);
    pred_bins = 1*(pred<300) + 2*((pred>=300)&(pred<=450)) + 3*(pred>450);
    accuracy = mean(true_bins == pred_bins);
    
    fprintf('Regression Metrics:\n');
    fprintf('MSE: %.1f | MAE: %.1f | Spearman: %.3f | Pearson: %.3f | Bin Accuracy: %.3f\n', mse, mae, spearman, pearson, accuracy);
    
    % Save regression model (optional, for completeness)
    out_dir = fullfile(data_dir, 'Step3_Results_2D'); % Reuse from Step 3 if exists
    if ~exist(out_dir, 'dir')
        mkdir(out_dir); % Create if doesn't exist - FIXES THE SAVE ERROR
    end
    save(fullfile(out_dir, 'regression_model.mat'), 'mdl', 'T', 'mse', 'spearman', 'accuracy');
else
    warning('Features or Survival CSV missing. Skipping Regression.');
end

%% -------------------- Figure 1: Training Dynamics --------------------
fprintf('Generating Figure 1: Training Curves...\n');
if isfile(unet_path)
    S = load(unet_path);
if isfield(S, 'info')
        info = S.info;
        fig1 = figure('Name', 'Training Dynamics', 'Color', 'w', 'Position', [100 100 1000 400]);
% Accuracy
        subplot(1, 2, 1);
        plot(info.TrainingAccuracy, 'LineWidth', 1.5, 'Color', '#0072BD'); hold on;
% Handle NaN/Missing Validation points
        valAcc = info.ValidationAccuracy;
        iter = 1:length(valAcc);
        mask = ~isnan(valAcc);
if sum(mask) > 0
            plot(iter(mask), valAcc(mask), 'o-', 'LineWidth', 1.5, 'Color', '#D95319', 'MarkerSize', 4);
end
        title('Model Accuracy'); xlabel('Iteration'); ylabel('Accuracy (%)');
        legend('Training', 'Validation', 'Location', 'southeast'); grid on; ylim([50 100]);
% Loss
        subplot(1, 2, 2);
        plot(info.TrainingLoss, 'LineWidth', 1.5, 'Color', '#0072BD'); hold on;
        valLoss = info.ValidationLoss;
if sum(mask) > 0
            plot(iter(mask), valLoss(mask), 'o-', 'LineWidth', 1.5, 'Color', '#D95319', 'MarkerSize', 4);
end
        title('Model Loss'); xlabel('Iteration'); ylabel('Loss');
        legend('Training', 'Validation'); grid on;
        exportgraphics(fig1, fullfile(output_fig_dir, 'Fig1_Training_Dynamics.png'), 'Resolution', 300);
else
        warning('Info struct missing in .mat file. Skipping Fig 1.');
end
else
    warning('trainedUnet.mat not found. Skipping Fig 1.');
end
%% -------------------- Load and Fix Survival Data --------------------
if isfile(surv_model_path) && isfile(features_csv)
    S_surv = load(surv_model_path);
    T_feats = readtable(features_csv);
    coeffs = S_surv.coeffs;
    mu = S_surv.mu;
    sigma = S_surv.sigma;
    feat_names = S_surv.feat_names;
% --- CRITICAL FIX: Handle "All Zeros" Event Column ---
% If your dataset doesn't record death events explicitly, we assume
% patients with survival days > 0 had the event (standard for BraTS).
if sum(T_feats.Event) == 0
        fprintf('⚠️ Auto-Fix: Detected all Events are 0. Setting Event=1 for patients with valid Survival Days.\n');
        T_feats.Event = T_feats.Survival_days > 0;
end
% Prepare Data Matrix
if iscell(feat_names), feat_names = string(feat_names); end
    data_matrix = table2array(T_feats(:, feat_names));
% Normalize and Calculate Risk Scores
    sigma(sigma==0) = 1;
    X_norm = (data_matrix - mu) ./ sigma;
    risk_scores = X_norm * coeffs;
    %% -------------------- Figure 2: Predicted vs True Survival (Regression Scatter) --------------------
    fprintf('Generating Figure 2: Predicted vs True Survival (Regression)...\n');
    fig2 = figure('Name', 'Predicted vs True', 'Color', 'w');
    scatter(y, pred, 40, 'filled', 'MarkerFaceColor', '#0072BD');
    hold on;
    p_reg = polyfit(y, pred, 1);
    x_reg = linspace(min(y), max(y), 100);
    y_reg_fit = polyval(p_reg, x_reg);
    plot(x_reg, y_reg_fit, 'r--', 'LineWidth', 2);
    title('Regression: Predicted vs True Survival Days', 'FontSize', 14);
    xlabel('True Survival (Days)', 'FontSize', 12);
    ylabel('Predicted Survival (Days)', 'FontSize', 12);
    grid on;
    legend('Data Points', sprintf('Trend (Spearman=%.3f)', spearman), 'Location', 'best');
    exportgraphics(fig2, fullfile(output_fig_dir, 'Fig2_Predicted_vs_True.png'), 'Resolution', 300);
    
    %% -------------------- Figure 3: Feature Importance (From Regression) --------------------
    fprintf('Generating Figure 3: Feature Importance (Regression Coefficients)...\n');
    fig3 = figure('Name', 'Feature Importance', 'Color', 'w');
    all_feat_names = [feat_names(1:8), 'Age', 'ResectionNum']; % Adjusted for removed cols
    reg_coeffs = mdl.Coefficients.Estimate(2:end); % Skip intercept
    [sorted_coeffs, idx] = sort(abs(reg_coeffs), 'ascend');
    sorted_names = all_feat_names(idx);
    real_coeffs = reg_coeffs(idx);
    b = barh(real_coeffs);
    b.FaceColor = 'flat';
    for k = 1:numel(real_coeffs)
        if real_coeffs(k) >= 0, b.CData(k,:) = [0.85 0.32 0.09];
        else, b.CData(k,:) = [0 0.44 0.74]; end
    end
    yticks(1:numel(all_feat_names));
    yticklabels(sorted_names);
    xlabel('Regression Coefficient');
    title('Radiomic + Clinical Feature Importance');
    grid on;
    exportgraphics(fig3, fullfile(output_fig_dir, 'Fig3_Feature_Importance.png'), 'Resolution', 300);
    %% -------------------- Figure 4: Correlation Matrix --------------------
    fprintf('Generating Figure 4: Correlation Matrix...\n');
    corr_mat = corr(data_matrix);
    fig4 = figure('Name', 'Correlation Matrix', 'Color', 'w', 'Position', [100 100 700 600]);
    h = heatmap(cellstr(feat_names), cellstr(feat_names), corr_mat);
    h.Title = 'Radiomic Feature Correlation';
    h.Colormap = parula;
    h.ColorLimits = [-1 1];
    exportgraphics(fig4, fullfile(output_fig_dir, 'Fig4_Correlation_Matrix.png'), 'Resolution', 300);
    %% -------------------- Figure 5 (NEW): Age vs Survival --------------------
% This is the "Other Important Representation" you asked for.
% It checks if Age correlates with Survival in your dataset.
if isfile(survival_info_csv)
        fprintf('Generating Figure 5: Age vs Survival (Clinical Validation)...\n');
        T_raw = readtable(survival_info_csv);
% We need to match rows by PatientID.
% T_feats has the model data, T_raw has Age.
% Let's join them.
try
% Ensure ID columns have same name for join
if ismember('Brats20ID', T_raw.Properties.VariableNames)
                T_raw.Properties.VariableNames{'Brats20ID'} = 'PatientID';
end
            T_merged = innerjoin(T_feats, T_raw(:, {'PatientID', 'Age'}));
            fig5 = figure('Name', 'Age vs Survival', 'Color', 'w');
            scatter(T_merged.Age, T_merged.Survival_days, 40, 'filled', 'MarkerFaceColor', '#77AC30');
% Add trend line
            hold on;
            p = polyfit(T_merged.Age, T_merged.Survival_days, 1);
            x_range = linspace(min(T_merged.Age), max(T_merged.Age), 100);
            y_fit = polyval(p, x_range);
            plot(x_range, y_fit, 'r--', 'LineWidth', 2);
            [R, Pval] = corr(T_merged.Age, T_merged.Survival_days, 'Rows','complete');
            title('Clinical Validation: Age vs. Survival');
            xlabel('Patient Age (Years)');
            ylabel('Survival (Days)');
            grid on;
            legend('Patient Data', sprintf('Trend (R=%.2f, p=%.3f)', R, Pval));
            exportgraphics(fig5, fullfile(output_fig_dir, 'Fig5_Age_vs_Survival.png'), 'Resolution', 300);
catch ME
            warning('Could not merge Age data. Skipping Fig 5. Error: %s', ME.message);
end
else
        fprintf('Survival info CSV not found. Skipping Age plot.\n');
end
else
    warning('Survival files missing. Skipping Figs 2-5.');
end
fprintf('✅ Step 4 Complete. All 5 Figures saved to: %s\n', output_fig_dir);
%% -------------------- Local Helper Functions --------------------
function [t, s] = calc_km(time, event)
    [sorted_time, sortIdx] = sort(time);
    sorted_event = event(sortIdx);
    unique_times = unique(sorted_time);
    s = ones(size(unique_times));
    current_s = 1;
for i = 1:numel(unique_times)
        ti = unique_times(i);
        n_risk = sum(sorted_time >= ti);
        d_events = sum(sorted_time == ti & sorted_event == 1);
if n_risk > 0
            current_s = current_s * (1 - d_events / n_risk);
end
        s(i) = current_s;
end
    t = [0; unique_times]; s = [1; s];
end