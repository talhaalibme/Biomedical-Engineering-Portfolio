%% AI-Driven Deep Brain Stimulation for Parkinson’s Tremor Suppression
% This script implements a five-phase project to detect Parkinson’s disease
% using voice data, analyze tremor severity, and simulate/optimize DBS effects.

%% ================================================
%% Phase 1: Data Exploration & Preprocessing
%% ================================================
clear; clc; close all;

%% 1) LOAD ALL DATASETS

% 1.1 Select and import voice CSVs
[pdFile, pdPath] = uigetfile('*.csv','Select pd_speech_features.csv (training data)');
[pkFile, pkPath] = uigetfile('*.data','Select parkinsons.data file');
if isequal(pdFile,0) || isequal(pkFile,0)
    error('You must select both pd_speech_features.csv and parkinsons.data');
end

% Read pd_speech
pd_full = fullfile(pdPath, pdFile);
pd_speech = readtable(pd_full, 'FileType','text','Delimiter',',');
if ~ismember('class', pd_speech.Properties.VariableNames)
    pd_speech.Properties.VariableNames{end} = 'class';
end

% Verify dataset size and class labels
disp('Size of pd_speech:');
disp(size(pd_speech));
disp('Unique labels in pd_speech.class before preprocessing:');
disp(unique(pd_speech.class));
if numel(unique(pd_speech.class)) < 2
    error('Loaded pd_speech has only one class. Please select the training dataset with both classes.');
end

% Read parkinsons
pk_full = fullfile(pkPath, pkFile);
parkinsons = readtable(pk_full, 'FileType','text','Delimiter',',');
if ~ismember('status', parkinsons.Properties.VariableNames)
    parkinsons.Properties.VariableNames{end} = 'status';
end

% Verify dataset size and status labels
disp('Size of parkinsons:');
disp(size(parkinsons));
disp('Unique labels in parkinsons.status before preprocessing:');
disp(unique(parkinsons.status));
if numel(unique(parkinsons.status)) < 2
    error('Loaded parkinsons has only one class. Please select a dataset with both classes.');
end

% 1.2 Select and import tremor files (.let, .rit)
tremorDir = uigetdir(pwd,'Select folder with tremor .let/.rit files');
letFiles = dir(fullfile(tremorDir,'*.let'));
ritFiles = dir(fullfile(tremorDir,'*.rit'));
allTremor = [letFiles; ritFiles];
if isempty(allTremor)
    warning('No tremor .let or .rit files found in the selected folder.');
end

tremorData = cell(numel(allTremor),1);
for k = 1:numel(allTremor)
    fp = fullfile(allTremor(k).folder, allTremor(k).name);
    raw = importdata(fp);
    tremorData{k} = raw;
    fprintf('Loaded tremor file: %s (%d samples)\n', allTremor(k).name, numel(raw));
end

% Estimate sampling rate based on first tremor file (assumed duration: 60s)
if ~isempty(tremorData)
    assumed_duration = 60; % seconds
    Fs = numel(tremorData{1}) / assumed_duration;
    fprintf('Estimated sampling rate: %.2f Hz\n', Fs);
else
    Fs = NaN;
end

%% 2) EXPLORE THE DATA

% 2.1 pd_speech: first 10 rows & variable names
disp('--- pd_speech first 10 rows ---');
disp(head(pd_speech,10));
disp('Variables in pd_speech:');
disp(pd_speech.Properties.VariableNames);
disp('Summary of pd_speech:'); summary(pd_speech);

% 2.2 parkinsons: first 10 rows & variable names
disp('--- parkinsons first 10 rows ---');
disp(head(parkinsons,10));
disp('Variables in parkinsons:'); disp(parkinsons.Properties.VariableNames);
disp('Summary of parkinsons:'); summary(parkinsons);

% 2.3 Visualize all tremor traces
if ~isempty(tremorData)
    numT = numel(tremorData);
    nCols = ceil(sqrt(numT)); nRows = ceil(numT/nCols);
    figure('Name','All Tremor Traces','NumberTitle','off');
    for k = 1:numT
        x = tremorData{k};
        t = (0:numel(x)-1)/Fs;
        subplot(nRows, nCols, k);
        plot(t, x);
        title(allTremor(k).name,'Interpreter','none');
        xlabel('Time (s)'); ylabel('Amplitude');
    end
    sgtitle('Tremor Velocity over Time for All Files');
end

%% 3) CLEAN THE DATA

% 3.1 Impute missing values with mean for numeric columns in pd_speech, excluding class
isNumeric_pd = varfun(@isnumeric, pd_speech, 'OutputFormat','uniform');
numVars_pd = pd_speech.Properties.VariableNames(isNumeric_pd);
numVars_pd = numVars_pd(~strcmp(numVars_pd, 'class')); % Exclude class
for v = numVars_pd
    col = pd_speech.(v{1});
    if any(ismissing(col))
        mean_val = mean(col, 'omitnan');
        col(ismissing(col)) = mean_val;
        pd_speech.(v{1}) = col;
    end
end

% Check for missing values in class column
if any(ismissing(pd_speech.class))
    error('Class column in pd_speech contains missing values. Please handle them manually.');
end

% Impute missing values with mean for numeric columns in parkinsons, excluding status
isNumeric_pk = varfun(@isnumeric, parkinsons, 'OutputFormat','uniform');
numVars_pk = parkinsons.Properties.VariableNames(isNumeric_pk);
numVars_pk = numVars_pk(~strcmp(numVars_pk, 'status')); % Exclude status
for v = numVars_pk
    col = parkinsons.(v{1});
    if any(ismissing(col))
        mean_val = mean(col, 'omitnan');
        col(ismissing(col)) = mean_val;
        parkinsons.(v{1}) = col;
    end
end

% Check for missing values in status column
if any(ismissing(parkinsons.status))
    error('Status column in parkinsons contains missing values. Please handle them manually.');
end

% Check unique labels after imputation
fprintf('Unique labels in pd_speech.class after imputation: %s\n', mat2str(unique(pd_speech.class)'));
fprintf('Unique labels in parkinsons.status after imputation: %s\n', mat2str(unique(parkinsons.status)'));

% 3.2 Outlier clipping for voice datasets
for v = numVars_pd
    pd_speech.(v{1}) = filloutliers(pd_speech.(v{1}), 'clip', 'ThresholdFactor', 3);
end
for v = numVars_pk
    parkinsons.(v{1}) = filloutliers(parkinsons.(v{1}), 'clip', 'ThresholdFactor', 3);
end

% 3.3 Tremor: interpolate NaNs
for k = 1:numel(tremorData)
    x = tremorData{k};
    if any(isnan(x))
        x = fillmissing(x,'linear');
        tremorData{k} = x;
    end
end

%% 4) NORMALIZE VOICE FEATURES

% pd_speech: standardize all numeric features except class
for v = numVars_pd
    pd_speech.(v{1}) = zscore(pd_speech.(v{1}));
end

% parkinsons: standardize all numeric features except status
for v = numVars_pk
    parkinsons.(v{1}) = zscore(parkinsons.(v{1}));
end

disp('✅ Phase 1 complete: data loaded, visualized, cleaned & normalized.');

%% ====================================================
%% Phase 2: Classification Models for Parkinson’s
%% ====================================================

%% A) pd_speech modeling with 5-fold CV

X_pd = pd_speech{:,1:end-1};
Y_pd = pd_speech.class;

% Check for multiple classes
if numel(unique(Y_pd)) < 2
    error('pd_speech has only one class after preprocessing.');
end

cv_pd = cvpartition(Y_pd, 'KFold', 5, 'Stratify', true);
pred_DT = []; pred_SVM = []; trueY = [];

for f = 1:cv_pd.NumTestSets
    tr = cv_pd.training(f); te = cv_pd.test(f);
    mdlDT = fitctree(X_pd(tr,:), Y_pd(tr), ...
             'PredictorNames', pd_speech.Properties.VariableNames(1:end-1));
    yDT   = predict(mdlDT, X_pd(te,:));
    mdlSVM = fitcsvm(X_pd(tr,:), Y_pd(tr), 'Standardize', true, 'KernelFunction', 'rbf');
    ySVM   = predict(mdlSVM, X_pd(te,:));
    pred_DT   = [pred_DT;   yDT];
    pred_SVM  = [pred_SVM;  ySVM];
    trueY     = [trueY;     Y_pd(te)];
end

% Initialize phase2_results structure
phase2_results = struct();

% Store Phase 2 results for pd_speech
phase2_results.pd_speech_DT_accuracy = mean(pred_DT==trueY)*100;
phase2_results.pd_speech_SVM_accuracy = mean(pred_SVM==trueY)*100;
fprintf('pd_speech DT CV Accuracy: %.2f%%\n', phase2_results.pd_speech_DT_accuracy);
fprintf('pd_speech SVM CV Accuracy: %.2f%%\n', phase2_results.pd_speech_SVM_accuracy);

% Full-data DT feature importance
mdlDT_all = fitctree(X_pd, Y_pd, 'PredictorNames', pd_speech.Properties.VariableNames(1:end-1));
imp_pd    = predictorImportance(mdlDT_all);
figure('Name', 'Feature Importance pd_speech', 'NumberTitle', 'off');
bar(imp_pd);
title('Feature Importance for pd_speech');
xlabel('Features'); ylabel('Importance');
set(gca,'XTick',1:numel(imp_pd),'XTickLabel',pd_speech.Properties.VariableNames(1:end-1),...
        'XTickLabelRotation',45);
phase2_results.pd_speech_feature_importance = imp_pd; % Save for later

disp('--- Completed pd_speech modeling ---');

%% B) parkinsons modeling with adaptive CV

% Ensure status is numeric 0/1
if iscell(parkinsons.status) || isstring(parkinsons.status)
    parkinsons.status = str2double(cellstr(parkinsons.status));
end

% Keep only rows labeled 0 or 1
mask = parkinsons.status==0 | parkinsons.status==1;
parkinsons = parkinsons(mask,:);

% Drop non-numerics (e.g. 'name') and isolate numeric features
isNumAll = varfun(@isnumeric, parkinsons, 'OutputFormat','uniform');
isNumAll(strcmp(parkinsons.Properties.VariableNames,'status')) = false;
numericFeats = parkinsons.Properties.VariableNames(isNumAll);

X_pk = parkinsons{:, numericFeats};
Y_pk = parkinsons.status;

% Check for multiple classes
if numel(unique(Y_pk)) < 2
    error('parkinsons has only one class after preprocessing.');
end

% Count minority class samples
n0   = sum(Y_pk==0);
n1   = sum(Y_pk==1);
nMin = min(n0,n1);

% Choose CV scheme
if nMin >= 5
    K = 5;
    cv_pk = cvpartition(Y_pk,'KFold',K,'Stratify',true);
    fprintf('Using %d-fold stratified CV\n', K);
elseif nMin >= 2
    K = nMin;
    cv_pk = cvpartition(Y_pk,'KFold',K,'Stratify',true);
    fprintf('Minority has %d samples: using %d-fold CV\n', nMin, K);
else
    cv_pk = cvpartition(Y_pk,'LeaveOut');
    fprintf('Only %d minority sample: using Leave-One-Out CV\n', nMin);
end

pDT = []; pSVM = []; truePK = [];
for f = 1:cv_pk.NumTestSets
    tr = cv_pk.training(f); te = cv_pk.test(f);
    mDT = fitctree(X_pk(tr,:), Y_pk(tr), 'PredictorNames', numericFeats);
    y1  = predict(mDT, X_pk(te,:));
    mSVM = fitcsvm(X_pk(tr,:), Y_pk(tr), 'Standardize', true, 'KernelFunction','rbf');
    y2   = predict(mSVM, X_pk(te,:));
    pDT   = [pDT; y1];
    pSVM  = [pSVM; y2];
    truePK= [truePK; Y_pk(te)];
end

% Store Phase 2 results for parkinsons
phase2_results.parkinsons_DT_accuracy = mean(pDT==truePK)*100;
phase2_results.parkinsons_SVM_accuracy = mean(pSVM==truePK)*100;
fprintf('parkinsons DT Accuracy: %.2f%%\n', phase2_results.parkinsons_DT_accuracy);
fprintf('parkinsons SVM Accuracy: %.2f%%\n', phase2_results.parkinsons_SVM_accuracy);

% Full-data DT feature importance
mDT_all_pk = fitctree(X_pk, Y_pk, 'PredictorNames', numericFeats);
imp_pk     = predictorImportance(mDT_all_pk);
figure('Name', 'Feature Importance parkinsons', 'NumberTitle', 'off');
bar(imp_pk);
title('Feature Importance for parkinsons');
xlabel('Features'); ylabel('Importance');
set(gca,'XTick',1:numel(imp_pk),'XTickLabel',numericFeats,'XTickLabelRotation',45);
phase2_results.parkinsons_feature_importance = imp_pk; % Save for later

disp('✅ Phase 2 complete: Models trained & evaluated with adaptive CV.');

%% ====================================================
%% Phase 3: Tremor Analysis & DBS Simulation
%% ====================================================
clearvars -except pd_speech parkinsons X_pd Y_pd X_pk Y_pk tremorData allTremor Fs phase2_results;
close all;

% Initialize phase3_results
phase3_results = struct();

% --- 1) Define condition map based on suffixes ---
conditionMap = containers.Map( ...
  {'ren','ref','ron','rof','r15of','r30of','r45of','r60of'}, ...
  {'ON_Med_ON','ON_Med_OFF','OFF_Med_ON','OFF_Med_OFF', ...
   'OFF+15min','OFF+30min','OFF+45min','OFF+60min'} ...
);

% Parse every file into (subject, condition)
N = numel(allTremor);
subjects  = cell(N,1);
conditions= cell(N,1);
for k = 1:N
    name = allTremor(k).name(1:end-4); % Remove .let or .rit extension
    tok = regexp(name, '^([gsv]\d+)(ren|ref|ron|rof|r\d+of)$', 'tokens');
    if ~isempty(tok) && iscell(tok) && ~isempty(tok{1})
        subjects{k} = tok{1}{1};
        suffix = tok{1}{2};
        if isKey(conditionMap, suffix)
            conditions{k} = conditionMap(suffix);
        else
            conditions{k} = 'Unknown';
        end
    else
        subjects{k} = sprintf('Unknown_%d', k);
        conditions{k} = 'Unknown';
    end
end

% Convert subjects to ensure all elements are strings
subjects = cellfun(@char, subjects, 'UniformOutput', false);
uniqueSubs = unique(subjects);

% --- 2) Plot DBS-OFF vs. ON for Med ON and Med OFF separately ---
groups = { {'ON_Med_ON','OFF_Med_ON'}, {'ON_Med_OFF','OFF_Med_OFF','OFF+15min','OFF+30min'} };
titles = {'Medication ON','Medication OFF & Recovery'};

for g = 1:2
    conds = groups{g};
    figure('Name',['Tremor: ' titles{g}],'NumberTitle','off');
    for s = 1:numel(uniqueSubs)
        sub = uniqueSubs{s};
        idxSub = strcmp(subjects, sub);
        idxPlot = ismember(conditions, conds) & idxSub;
        if ~any(idxPlot), continue; end

        subplot(numel(uniqueSubs),1,s); hold on;
        cmap = lines(numel(conds));
        for c = 1:numel(conds)
            idxC = idxSub & strcmp(conditions, conds{c});
            if any(idxC)
                kf = find(idxC,1);
                sig = tremorData{kf};
                t   = (0:numel(sig)-1)/Fs;
                plot(t, sig, 'Color', cmap(c,:), 'DisplayName', conds{c});
            end
        end
        hold off;
        legend('Location','best','Interpreter','none');
        title(['Subject ' sub],'Interpreter','none');
        xlabel('Time (s)'); ylabel('Velocity'); grid on;
    end
    sgtitle(['Tremor Comparison — ' titles{g}]);
end

% --- 3) Compute RMS & Dominant Frequency for every file ---
rmsVals  = zeros(N,1);
domFreqs = zeros(N,1);
for k = 1:N
    sig = tremorData{k};
    rmsVals(k) = sqrt(mean(sig.^2));
    L = numel(sig);
    Yf = fft(sig);
    P2 = abs(Yf/L);
    P1 = P2(1:floor(L/2)+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(0:floor(L/2))/L;
    [~,pi] = max(P1);
    domFreqs(k) = f(pi);
end

T = table(subjects, conditions, rmsVals, domFreqs, 'VariableNames',{'Subject','Condition','RMS','DomFreq'});

% --- 4) Bar charts of metrics by condition ---
condsList = unique(conditions);
M_rms  = zeros(numel(uniqueSubs), numel(condsList));
M_freq = M_rms;
for i = 1:numel(uniqueSubs)
    for j = 1:numel(condsList)
        sel = strcmp(T.Subject, uniqueSubs{i}) & strcmp(T.Condition, condsList{j});
        if any(sel)
            M_rms(i,j)  = mean(T.RMS(sel));
            M_freq(i,j) = mean(T.DomFreq(sel));
        else
            M_rms(i,j)  = NaN;
            M_freq(i,j) = NaN;
        end
    end
end

% Store Phase 3 results
phase3_results.RMS_by_condition = M_rms;
phase3_results.DomFreq_by_condition = M_freq;
phase3_results.unique_subjects = uniqueSubs;
phase3_results.conditions_list = condsList;

figure('Name','RMS by Condition','NumberTitle','off');
bar(M_rms,'grouped');
xticks(1:numel(uniqueSubs)); xticklabels(uniqueSubs);
legend(condsList,'Location','best');
xlabel('Subject'); ylabel('RMS'); title('RMS Amplitude by Condition');
xtickangle(45);

figure('Name','Dominant Freq by Condition','NumberTitle','off');
bar(M_freq,'grouped');
xticks(1:numel(uniqueSubs)); xticklabels(uniqueSubs);
legend(condsList,'Location','best');
xlabel('Subject'); ylabel('Frequency (Hz)'); title('Dominant Frequency by Condition');
xtickangle(45);

disp('✅ Phase 3 complete: visualizations and metrics ready.');

% --- 5) DBS Simulation in Simulink ---
% Define tremor data for simulation
off_name = 'g1ron.let'; % DBS-OFF condition
on_name = 'g1ren.let';  % DBS-ON condition
idx_off = find(strcmp({allTremor.name}, off_name), 1);
idx_on = find(strcmp({allTremor.name}, on_name), 1);

if isempty(idx_off) || isempty(idx_on)
    error('Could not find tremor files "%s" or "%s" in allTremor.', off_name, on_name);
end

raw_off = tremorData{idx_off}; % Tremor data when DBS is OFF
raw_on = tremorData{idx_on};   % Tremor data when DBS is ON
t_off = (0:length(raw_off)-1)' / Fs;
t_on = (0:length(raw_on)-1)' / Fs;

% Create timeseries objects
ts_off = timeseries(raw_off, t_off);
ts_on = timeseries(raw_on, t_on);

% Create Simulink model programmatically
mdl = 'dbs_simulation';
if bdIsLoaded(mdl)
    close_system(mdl, 0);
end

load_system('simulink');
new_system(mdl);

% Add blocks
add_block('simulink/Sources/From Workspace', [mdl '/TremorInputOff'], ...
          'VariableName', 'ts_off', 'Position', [50, 50, 150, 90]);
add_block('simulink/Sources/From Workspace', [mdl '/TremorInputOn'], ...
          'VariableName', 'ts_on', 'Position', [50, 150, 150, 190]);
add_block('simulink/Sources/Constant', [mdl '/DBSAmplitude'], ...
          'Value', '0.5', 'Position', [200, 100, 240, 140]); % DBS effect amplitude
add_block('simulink/Math Operations/Sum', [mdl '/ApplyDBS'], ...
          'Inputs', '-+', 'Position', [300, 50, 340, 90]); % Subtract DBS effect
add_block('simulink/Sinks/To Workspace', [mdl '/OutputTremor'], ...
          'VariableName', 'output_tremor', 'Position', [400, 50, 500, 90]);

% Connect blocks
add_line(mdl, 'TremorInputOff/1', 'ApplyDBS/1');
add_line(mdl, 'DBSAmplitude/1', 'ApplyDBS/2');
add_line(mdl, 'ApplyDBS/1', 'OutputTremor/1');

% Configure simulation parameters
set_param(mdl, 'StopTime', num2str(max(t_off)));
set_param(mdl, 'Solver', 'FixedStepDiscrete');
set_param(mdl, 'FixedStep', num2str(1/Fs));
save_system(mdl);

% Run simulation
simIn = Simulink.SimulationInput(mdl);
simIn = simIn.setVariable('ts_off', ts_off);
simIn = simIn.setVariable('ts_on', ts_on);
simOut = sim(simIn);

% Extract simulated tremor
output_tremor = simOut.get('output_tremor').Data;

% Ensure consistent lengths
if length(output_tremor) ~= length(raw_off)
    output_tremor = interp1(linspace(0, max(t_off), length(output_tremor)), output_tremor, t_off);
end

% Calculate tremor reduction
rms_sim = sqrt(mean(output_tremor.^2));
rms_off = sqrt(mean(raw_off.^2));
rms_on = sqrt(mean(raw_on.^2));
reduction = (rms_off - rms_sim) / rms_off * 100;
phase3_results.tremor_reduction = reduction;
fprintf('Tremor reduction with simulated DBS: %.2f%%\n', reduction);

% Plot comparison
figure('Name', 'DBS Simulation Results', 'NumberTitle', 'off');
subplot(3,1,1);
plot(t_off, raw_off, 'b');
title('DBS-OFF Tremor'); xlabel('Time (s)'); ylabel('Velocity');
subplot(3,1,2);
plot(t_on, raw_on, 'g');
title('DBS-ON Tremor'); xlabel('Time (s)'); ylabel('Velocity');
subplot(3,1,3);
plot(t_off, output_tremor, 'r');
title('Simulated DBS Tremor'); xlabel('Time (s)'); ylabel('Velocity');

disp('✅ Phase 3 fully complete: DBS simulation executed.');

%% ====================================================
%% Phase 4: Systematic Model Comparison & Selection
%% ====================================================
rng(42);  % reproducibility

% Initialize phase4_results structure
phase4_results = struct();

% 1) Define models
models = {
    struct('name','DecisionTree','train',@(X,Y) fitctree(X,Y)), 
    struct('name','SVM_RBF',    'train',@(X,Y) fitcsvm(X,Y,...
                           'KernelFunction','rbf','Standardize',true,...
                           'KernelScale','auto','BoxConstraint',1,'ClassNames',[0,1])), 
    struct('name','Logistic',   'train',@(X,Y) fitclinear(X,Y,...
                           'Learner','logistic','Regularization','lasso','Lambda',0.01)), 
    struct('name','Ensemble',   'train',@(X,Y) fitcensemble(X,Y,'Method','Bag'))
};

% 2) Wrap datasets
datasets = {
    struct('tag','pd_speech','X',X_pd,'Y',Y_pd,'folds',5), 
    struct('tag','parkinsons','X',X_pk,'Y',Y_pk,'folds',[])
};

% 3) Manual K-fold loop for all models
for d = 1:numel(datasets)
    Xd = datasets{d}.X;
    Yd = datasets{d}.Y;
    tag = datasets{d}.tag;

    if ~isempty(datasets{d}.folds)
        cvp = cvpartition(Yd,'KFold',datasets{d}.folds,'Stratify',true);
    else
        nMin = min(sum(Yd==0), sum(Yd==1));
        if nMin >= 5
            cvp = cvpartition(Yd,'KFold',5,'Stratify',true);
        else
            cvp = cvpartition(Yd,'LeaveOut');
        end
    end

    fprintf('\n--- CV Results for %s ---\n', tag);
    metrics    = {'Accuracy','Precision','Recall','F1','AUC'};
    modelNames = cellfun(@(s) s.name, models, 'UniformOutput', false);
    metricsTable = array2table(zeros(numel(models),numel(metrics)), ...
        'RowNames', modelNames, 'VariableNames', metrics);

    for m = 1:numel(models)
        allPred  = nan(size(Yd));
        allScore = nan(numel(Yd),2);
        for f = 1:cvp.NumTestSets
            trIdx = cvp.training(f);
            teIdx = cvp.test(f);
            mdl = models{m}.train(Xd(trIdx,:), Yd(trIdx));
            [yhat, sc] = predict(mdl, Xd(teIdx,:));
            if size(sc,2)==1
                sc = [1-sc, sc];
            end
            allPred(teIdx)     = yhat;
            allScore(teIdx, :) = sc;
        end
        cm = confusionmat(Yd, allPred, 'Order',[0,1]);
        TP = cm(2,2); TN = cm(1,1); FP = cm(1,2); FN = cm(2,1);
        metricsTable.Accuracy(m)  = (TP+TN)/sum(cm(:));
        metricsTable.Precision(m) = TP/(TP+FP+eps);
        metricsTable.Recall(m)    = TP/(TP+FN+eps);
        metricsTable.F1(m)        = 2*(metricsTable.Precision(m)*metricsTable.Recall(m))/(metricsTable.Precision(m)+metricsTable.Recall(m)+eps);
        if numel(unique(Yd))>=2
            [~,~,~,metricsTable.AUC(m)] = perfcurve(Yd, allScore(:,2), 1);
        else
            metricsTable.AUC(m) = NaN;
        end
    end
    disp(metricsTable);
    if strcmp(tag, 'pd_speech')
        phase4_results.pd_speech_metrics = metricsTable;
    else
        phase4_results.parkinsons_metrics = metricsTable;
    end
end

% 4) Hyperparameter tuning SVM on Parkinsons
u = unique(Y_pk);
if numel(u) < 2
    warning('Y_pk has only one class (%d). Skipping hyperparameter tuning.', u);
else
    fprintf('\n--- Hyperparameter Tuning (SVM) on parkinsons ---\n');
    svmOpt = fitcsvm( X_pk, Y_pk, ...
        'KernelFunction','rbf', ...
        'Standardize',true, ...
        'ClassNames',[0 1], ...
        'OptimizeHyperparameters',{'BoxConstraint','KernelScale'}, ...
        'HyperparameterOptimizationOptions',struct( ...
            'AcquisitionFunctionName','expected-improvement-plus', ...
            'ShowPlots',true,'UseParallel',false,'Verbose',0) );  % Disable parallel
    bestP = svmOpt.ModelParameters;
    fprintf('Best BoxConstraint=%.3g, KernelScale=%.3g\n', bestP.BoxConstraint, bestP.KernelScale);
    phase4_results.svm_best_params = bestP;
end

% 5) Stratified hold-out (80/20) on Parkinsons
if ~exist('bestP','var')
    bestP.BoxConstraint = 1;
    bestP.KernelScale   = 'auto';
end

fprintf('\n--- Hold-Out Validation (SVM) on parkinsons ---\n');
c0 = find(Y_pk==0); c1 = find(Y_pk==1);
n0 = numel(c0); n1 = numel(c1);
n0t = floor(0.2*n0); n1t = floor(0.2*n1);
idxTest  = [c0(randperm(n0,n0t)); c1(randperm(n1,n1t))];
idxTrain = setdiff(1:numel(Y_pk), idxTest);

Xtr = X_pk(idxTrain,:);  Ytr = Y_pk(idxTrain);
Xte = X_pk(idxTest,:);   Yte = Y_pk(idxTest);

svmFinal = fitcsvm( Xtr, Ytr, ...
    'KernelFunction','rbf', ...
    'BoxConstraint', bestP.BoxConstraint, ...
    'KernelScale',   bestP.KernelScale, ...
    'Standardize',   true );

pred_te = predict(svmFinal, Xte);
cm_te = confusionmat(Yte, pred_te, 'Order',[0,1]);

TN = cm_te(1,1); FP = cm_te(1,2); FN = cm_te(2,1); TP = cm_te(2,2);

accuracy    = (TP + TN) / sum(cm_te(:));
precision   = TP / (TP + FP + eps);
recall      = TP / (TP + FN + eps);
f1_score    = 2 * (precision * recall) / (precision + recall + eps);

phase4_results.holdout_accuracy = accuracy * 100;
phase4_results.holdout_precision = precision * 100;
phase4_results.holdout_recall = recall * 100;
phase4_results.holdout_f1_score = f1_score * 100;
phase4_results.holdout_confusion_matrix = cm_te;

fprintf('\nHold-Out SVM Results:\n');
fprintf('  Accuracy : %.2f%%\n', phase4_results.holdout_accuracy);
fprintf('  Precision: %.2f%%\n', phase4_results.holdout_precision);
fprintf('  Recall   : %.2f%%\n', phase4_results.holdout_recall);
fprintf('  F1 Score : %.2f%%\n', phase4_results.holdout_f1_score);
disp('Confusion matrix (rows=true, cols=pred):');
disp(phase4_results.holdout_confusion_matrix);

save('phase4_results.mat', 'phase4_results', 'T');

disp('✅ Phase 4 complete: Model comparison and selection done.');

%% ====================================================
%% Phase 5: Integration and Validation
%% ====================================================

load('phase4_results.mat');

fprintf('\n--- Summary of PD Detection ---\n');
fprintf('pd_speech DT CV Accuracy: %.2f%%\n', phase2_results.pd_speech_DT_accuracy);
fprintf('pd_speech SVM CV Accuracy: %.2f%%\n', phase2_results.pd_speech_SVM_accuracy);
fprintf('parkinsons DT Accuracy: %.2f%%\n', phase2_results.parkinsons_DT_accuracy);
fprintf('parkinsons SVM Accuracy: %.2f%%\n', phase2_results.parkinsons_SVM_accuracy);
fprintf('Hold-out SVM Accuracy: %.2f%%\n', phase4_results.holdout_accuracy);

fprintf('\n--- Summary of Tremor Analysis ---\n');
disp('RMS and Dominant Frequency by Condition:');
disp(T);  % Assuming T is the table with RMS and DomFreq

fprintf('\n--- Summary of DBS Simulation ---\n');
fprintf('Tremor reduction with simulated DBS: %.2f%%\n', phase3_results.tremor_reduction);

disp('✅ Phase 5 complete: Project summary and validation.');