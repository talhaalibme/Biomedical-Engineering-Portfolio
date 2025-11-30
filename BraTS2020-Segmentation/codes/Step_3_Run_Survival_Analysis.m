%% Step_3_Run_Survival_2D_Final.m
% Standalone script: 2D slice segmentation -> feature extraction -> survival modeling
% Cleaned: all helper functions are at the bottom and properly closed with 'end'.

clearvars; clc; close all;
fprintf('Starting Step 3 (2D slice-by-slice inference) - Feature extraction + Survival modeling\n');

%% -------------------- User config --------------------
data_root     = 'BraTS2020_TrainingData';
patients_dir  = fullfile(data_root, 'MICCAI_BraTS2020_TrainingData');
preproc_imgs  = fullfile(data_root, 'Preprocessed_Data', 'Images');
name_map_csv  = fullfile(data_root, 'name_mapping.csv');
name_map_xlsx = fullfile(data_root, 'name_mapping.xlsx');
surv_csv      = fullfile(data_root, 'survival_info.csv');
surv_xlsx     = fullfile(data_root, 'survival_info.xlsx');
unet_file_default = fullfile(data_root, 'Trained_Unet', 'trainedUnet.mat');

maxPatients = [];      % set to a number to limit processing
VERBOSE = true;        % set to false to reduce printing

%% -------------------- Load mapping --------------------
if isfile(name_map_csv)
    name_map = readtable(name_map_csv, 'PreserveVariableNames', true);
elseif isfile(name_map_xlsx)
    name_map = readtable(name_map_xlsx, 'PreserveVariableNames', true);
else
    error('name_mapping not found (expected %s or %s)', name_map_csv, name_map_xlsx);
end

if isfile(surv_csv)
    surv_map = readtable(surv_csv, 'PreserveVariableNames', true);
elseif isfile(surv_xlsx)
    surv_map = readtable(surv_xlsx, 'PreserveVariableNames', true);
else
    surv_map = table();
end

patient_id_col = detect_patient_id_column(name_map);
raw_ids = name_map.(patient_id_col);
if iscell(raw_ids)
    patient_ids_all = cellfun(@(x) char(string(x)), raw_ids, 'UniformOutput', false);
else
    patient_ids_all = cellfun(@(x) char(string(x)), num2cell(raw_ids), 'UniformOutput', false);
end
keep_mask = cellfun(@(s) ~isempty(strtrim(s)) && ~strcmpi(strtrim(s),'NA') && ~strcmpi(strtrim(s),'NaN'), patient_ids_all);
patient_ids_all = patient_ids_all(keep_mask);
if ~isempty(maxPatients)
    patient_ids_all = patient_ids_all(1:min(maxPatients, numel(patient_ids_all)));
end

%% -------------------- Load trained U-Net --------------------
unet_file = unet_file_default;
if ~isfile(unet_file)
    error('U-Net file not found: %s', unet_file);
end

S = load(unet_file);
net = [];
vars = fieldnames(S);
for k = 1:numel(vars)
    v = S.(vars{k});
    if isa(v, 'DAGNetwork') || isa(v, 'SeriesNetwork') || isa(v, 'dlnetwork')
        net = v;
        break;
    end
end
if isempty(net) && isfield(S, 'net')
    net = S.net;
end
if isempty(net)
    % try any field that contains 'unet' or 'net'
    for k = 1:numel(vars)
        if contains(vars{k}, 'unet', 'IgnoreCase', true) || contains(vars{k}, 'net', 'IgnoreCase', true)
            net = S.(vars{k});
            break;
        end
    end
end
if isempty(net)
    error('No network object found inside MAT file. Inspect %s', unet_file);
end

% input size
try
    inputSize = net.Layers(1).InputSize;
catch
    if isprop(net, 'InputSize')
        inputSize = net.InputSize;
    else
        inputSize = [128 128 1];
    end
end
netH = inputSize(1); netW = inputSize(2); netC = inputSize(3);
if VERBOSE
    fprintf('Network expects input size [%d %d %d]\n', netH, netW, netC);
end

%% -------------------- Survival map detection --------------------
if ~isempty(surv_map)
    surv_id_col = detect_patient_id_column(surv_map);
    if ~isempty(surv_id_col)
        surv_id_strings = string(surv_map.(surv_id_col));
    else
        surv_id_strings = string(surv_map{:,1});
    end
    survival_col = detect_column_by_keywords(surv_map, {'survival','os','time','days','followup'});
    event_col = detect_column_by_keywords(surv_map, {'event','status','dead','censor'});
else
    surv_id_strings = string([]);
    survival_col = '';
    event_col = '';
end

%% -------------------- Containers --------------------
feat_names = {'Vol_voxels','Vol_mm3','BBoxVol_voxels','BBoxVol_mm3','MeanInt','MinInt','MaxInt','VarInt','SkewInt','KurtInt'};
all_features = zeros(0, numel(feat_names));
all_ids = {};
all_surv = [];
all_event = [];

%% -------------------- Main Loop --------------------
n_processed = 0;
tstart = tic;
for i = 1:numel(patient_ids_all)
    pid = patient_ids_all{i};
    if isempty(pid), continue; end
    if VERBOSE, fprintf('\n[%d/%d] Patient: %s\n', i, numel(patient_ids_all), pid); end

    % find flair file
    patient_folder = find_patient_folder(patients_dir, pid);
    flair_file = '';
    if ~isempty(patient_folder)
        flair_file = find_flair_file(patient_folder, pid);
    end

    % fallback: preprocessed PNGs
    use_preproc_png = false;
    png_files = {};
    if isempty(flair_file) && isfolder(preproc_imgs)
        d = dir(fullfile(preproc_imgs, ['*' pid '*']));
        d = d(~[d.isdir]);
        if ~isempty(d)
            maskpng = endsWith({d.name}, '.png', 'IgnoreCase', true) | endsWith({d.name}, '.jpg', 'IgnoreCase', true);
            d = d(maskpng);
            if ~isempty(d)
                png_files = natsortfiles({d.name});
                use_preproc_png = true;
            end
        end
    end

    if isempty(flair_file) && ~use_preproc_png
        if VERBOSE, fprintf('  No flair or preprocessed slices for %s - skipping\n', pid); end
        continue;
    end

    % read volume
    if ~isempty(flair_file)
        try
            info = niftiinfo(flair_file);
            vol = double(niftiread(info));
            vol = squeeze(vol);
            if ndims(vol) > 3
                vol = squeeze(vol(:,:,:,1));
            end
            [H, W, Z] = size(vol);
            vox_mm = try_get_pixdim(info);
        catch ME
            if VERBOSE, fprintf('  Error reading NIfTI: %s\n', ME.message); end
            continue;
        end
    else
        Z = numel(png_files);
        sampleI = imread(fullfile(preproc_imgs, png_files{1}));
        if ndims(sampleI) == 3, sampleI = rgb2gray(sampleI); end
        [H, W] = size(sampleI);
        vol = zeros(H, W, Z);
        for z = 1:Z
            I = imread(fullfile(preproc_imgs, png_files{z}));
            if ndims(I) == 3, I = rgb2gray(I); end
            vol(:,:,z) = double(I);
        end
        vox_mm = [1 1 1];
    end

    % segmentation slice-by-slice
    pred_mask = false(H, W, Z);
    for z = 1:Z
        slice = vol(:,:,z);
        if ~any(slice(:)), continue; end
        % normalization used at inference (simple)
        slice_resized = imresize(slice, [netH netW]);
        Iin = im2single(slice_resized);
        if netC > 1
            % replicate grayscale across channels
            warn_once(sprintf('%s_netC%d', pid, netC), VERBOSE, netC);
            Iin = repmat(Iin, [1 1 netC]);
        end
        % semantic segmentation
        try
            C = semanticseg(Iin, net);
        catch
            try
                C = semanticseg(im2uint8(Iin), net);
            catch ME
                if VERBOSE, fprintf('  semanticseg failed on slice %d: %s\n', z, ME.message); end
                continue;
            end
        end

        % convert output to logical mask robustly
        mask2d = segOutputToMask(C);

        mask_up = imresize(double(mask2d), [H, W], 'nearest') > 0.5;
        pred_mask(:,:,z) = mask_up;

        if VERBOSE
            try
                uniq = unique(C(:));
                fprintf('  slice %d unique labels: %s\n', z, strjoin(string(uniq(:)')));
            catch
                fprintf('  slice %d unique labels: (unable to display)\n', z);
            end
        end
    end

    % postprocess: largest component
    cc = bwconncomp(pred_mask, 26);
    if cc.NumObjects == 0
        vol_vox = 0; meanI = 0; minI = 0; maxI = 0; varI = 0; skewI = 0; kurtI = 0; bbox_vol_vox = 0;
    else
        sizes = cellfun(@numel, cc.PixelIdxList);
        [~, idxmax] = max(sizes);
        mask_largest = false(size(pred_mask));
        mask_largest(cc.PixelIdxList{idxmax}) = true;
        vals = vol(mask_largest);
        if isempty(vals)
            vol_vox = 0; meanI = 0; minI = 0; maxI = 0; varI = 0; skewI = 0; kurtI = 0;
        else
            vol_vox = numel(vals);
            meanI = mean(vals);
            minI = min(vals);
            maxI = max(vals);
            varI = var(vals);
            skewI = skewness(vals);
            kurtI = kurtosis(vals);
        end
        stats = regionprops3(mask_largest, 'BoundingBox');
        if isempty(stats)
            bbox_vol_vox = 0;
        else
            bb = stats.BoundingBox(1,:);
            bbox_vol_vox = bb(4) * bb(5) * bb(6);
        end
    end

    vox_vol_mm3 = prod(vox_mm);
    vol_mm3 = vol_vox * vox_vol_mm3;
    bbox_mm3 = bbox_vol_vox * vox_vol_mm3;
    feats = [vol_vox, vol_mm3, bbox_vol_vox, bbox_mm3, meanI, minI, maxI, varI, skewI, kurtI];

    all_features(end+1, :) = feats;
    all_ids{end+1} = pid;

    % survival lookup
    surv_time = NaN; surv_event = NaN;
    if ~isempty(surv_map)
        idx_match = find(contains(surv_id_strings, pid, 'IgnoreCase', true), 1);
        if isempty(idx_match)
            d = regexp(pid, '\d+', 'match');
            if ~isempty(d)
                idx_match = find(contains(surv_id_strings, d{end}), 1);
            end
        end
        if ~isempty(idx_match)
            if ~isempty(survival_col)
                try surv_time = double(surv_map.(survival_col)(idx_match)); catch, surv_time = NaN; end
            end
            if ~isempty(event_col)
                try surv_event = double(surv_map.(event_col)(idx_match)); catch, surv_event = NaN; end
            end
        end
    end

    if isempty(surv_event) || isnan(surv_event)
        all_event(end+1, 1) = 0;
    else
        all_event(end+1, 1) = surv_event;
    end
    all_surv(end+1, 1) = surv_time;

    n_processed = n_processed + 1;
    if VERBOSE
        fprintf('  Processed %d: %s  (vol vox: %d)  surv: %g  event: %g\n', n_processed, pid, vol_vox, surv_time, all_event(end));
    end
end

elapsed = toc(tstart);
fprintf('\nFinished feature extraction for %d patients in %.1f seconds.\n', n_processed, elapsed);

%% -------------------- Build table & Cox model --------------------
T = array2table(all_features, 'VariableNames', feat_names);
T.PatientID = all_ids';
T.Survival_days = all_surv;
T.Event = all_event;

valid_idx = ~isnan(T.Survival_days) & (T.Survival_days > 0);
T_valid = T(valid_idx, :);

out_dir = fullfile(data_root, 'Step3_Results_2D');
if ~isfolder(out_dir)
    mkdir(out_dir);
end

writetable(T, fullfile(out_dir, 'features_table_all.csv'));

if isempty(T_valid)
    warning('No valid survival records found. Saved features only.');
    save(fullfile(out_dir, 'features_only.mat'), 'T');
    return;
end

X = table2array(T_valid(:, feat_names));
mu = mean(X, 1);
sigma = std(X, [], 1);
sigma(sigma == 0) = 1;
Xnorm = (X - mu) ./ sigma;
Ttime = T_valid.Survival_days;
Censoring = ~logical(T_valid.Event);

coeffs = [];
model_out = struct();
try
    [beta, ~, stats] = coxphfit(Xnorm, Ttime, 'Censoring', Censoring);
    coeffs = beta;
    model_out.type = 'coxphfit';
    model_out.beta = beta;
    model_out.stats = stats;
catch ME
    warning('Cox fit failed: %s', ME.message);
    coeffs = zeros(size(X, 2), 1);
end

risk_scores = Xnorm * coeffs;
EventBool = ~Censoring;
cidx = concordance_index(Ttime, EventBool, -risk_scores);
fprintf('Final C-Index: %.4f\n', cidx);

save(fullfile(out_dir, 'survival_model.mat'), 'model_out', 'coeffs', 'feat_names', 'cidx', 'T', 'mu', 'sigma');
writetable(T_valid, fullfile(out_dir, 'features_table_for_model.csv'));
fprintf('Saved results to %s\n', out_dir);

%% -------------------- Helper functions (all ended with "end") --------------------

function colname = detect_patient_id_column(tbl)
vars = tbl.Properties.VariableNames;
bestScore = -inf;
colname = vars{1};
n = height(tbl);
for k = 1:numel(vars)
    col = tbl.(vars{k});
    try
        colStr = string(col);
    catch
        continue;
    end
    maskNonEmpty = ~ismissing(colStr) & strlength(strtrim(colStr)) > 0;
    fracNonEmpty = sum(maskNonEmpty) / max(1, n);
    fracBraTS = sum(contains(colStr, {'BraTS', 'Brats', 'TCIA', 'BRATS'}, 'IgnoreCase', true)) / max(1, n);
    score = 2 * fracBraTS + 0.5 * fracNonEmpty;
    if score > bestScore
        bestScore = score;
        colname = vars{k};
    end
end
end

function colname = detect_column_by_keywords(tbl, keywords)
vars = tbl.Properties.VariableNames;
colname = '';
for k = 1:numel(vars)
    if any(contains(vars{k}, keywords, 'IgnoreCase', true))
        colname = vars{k};
        return;
    end
end
for k = 1:numel(vars)
    col = tbl.(vars{k});
    if isnumeric(col)
        continue;
    end
    s = string(col);
    if any(contains(s, keywords, 'IgnoreCase', true))
        colname = vars{k};
        return;
    end
end
end

function folder = find_patient_folder(root, pid)
candidates = {
    fullfile(root, pid), ...
    fullfile(root, ['BraTS20_Training_' pid]), ...
    fullfile(root, ['BraTS20_Training_' sprintf('%03d', str2double(pid))]), ...
    fullfile(root, ['BraTS20_Training_' regexprep(pid, '[^0-9]', '')])
};
folder = '';
for k = 1:numel(candidates)
    if isfolder(candidates{k})
        folder = candidates{k};
        return;
    end
end
d = dir(fullfile(root, ['*' pid '*']));
if ~isempty(d)
    folder = fullfile(d(1).folder, d(1).name);
end
end

function fpath = find_flair_file(pfolder, pid)
patterns = {
    sprintf('%s_flair.nii', pid), sprintf('%s_flair.nii.gz', pid), ...
    '*_flair*.nii', '*_flair*.nii.gz', 'flair.nii', 'FLAIR.nii', 'flair.nii.gz', 'FLAIR.nii.gz', ...
    '*FLAIR*.*', '*flair*.*'
};
fpath = '';
for pp = 1:numel(patterns)
    c = dir(fullfile(pfolder, patterns{pp}));
    if ~isempty(c)
        fpath = fullfile(c(1).folder, c(1).name);
        return;
    end
end
end

function vox = try_get_pixdim(info)
try
    pixdim = info.PixelDimensions;
    if numel(pixdim) < 3
        pixdim = [pixdim, 1];
    end
    vox = pixdim;
catch
    vox = [1 1 1];
end
end

function out = natsortfiles(in)
[~, idx] = sort_nat(in);
out = in(idx);
end

function [S, idx] = sort_nat(C)
tokens = cellfun(@(s) regexp(s, '(\d+)', 'match'), C, 'UniformOutput', false);
maxTokens = max(cellfun(@numel, tokens));
numVec = zeros(numel(C), maxTokens);
for i = 1:numel(C)
    t = tokens{i};
    for j = 1:maxTokens
        if j <= numel(t)
            numVec(i, j) = str2double(t{j});
        else
            numVec(i, j) = Inf;
        end
    end
end
[~, idx] = sortrows(numVec);
S = C(idx);
end

function c = concordance_index(T, Event, score)
n = numel(T);
concordant = 0;
permissible = 0;
for i = 1:n-1
    for j = i+1:n
        ti = T(i); tj = T(j); ei = Event(i); ej = Event(j);
        if ei == 1 && ej == 1
            permissible = permissible + 1;
            if score(i) == score(j)
                concordant = concordant + 0.5;
            elseif (ti < tj && score(i) > score(j)) || (tj < ti && score(j) > score(i))
                concordant = concordant + 1;
            end
        elseif ei == 1 && ej == 0
            if ti <= tj
                permissible = permissible + 1;
                if score(i) == score(j)
                    concordant = concordant + 0.5;
                elseif (ti < tj && score(i) > score(j))
                    concordant = concordant + 1;
                end
            end
        elseif ei == 0 && ej == 1
            if tj <= ti
                permissible = permissible + 1;
                if score(i) == score(j)
                    concordant = concordant + 0.5;
                elseif (tj < ti && score(j) > score(i))
                    concordant = concordant + 1;
                end
            end
        end
    end
end
if permissible == 0
    c = NaN;
else
    c = concordant / permissible;
end
end

function printed = warn_once(key, verbose, netC)
persistent warnedKeys
if isempty(warnedKeys)
    warnedKeys = containers.Map('KeyType', 'char', 'ValueType', 'logical');
end
if ~ischar(key)
    key = char(string(key));
end
if ~isKey(warnedKeys, key)
    warnedKeys(key) = true;
    if verbose
        fprintf('  Warning: network expects %d channels, replicating grayscale to match.\n', netC);
    end
    printed = true;
else
    printed = false;
end
end

function mask = segOutputToMask(C)
% Robust conversion of semanticseg output to logical mask
if isempty(C)
    mask = false(1,1);
    return;
end
try
    if iscategorical(C)
        cats = categories(C);
        % if there is an explicit 'background' category, treat it as bg
        if any(ismember(lower(cats), lower("background")))
            mask = ~(C == categorical("background"));
        else
            % treat entries equal to 'tumor' or '1' as positive
            mask = (C == categorical("tumor")) | (C == categorical("1")) | (C == categorical("Tumor"));
        end
    elseif islogical(C)
        mask = C(:,:,1);
    elseif isnumeric(C)
        mask = C(:,:,1) ~= 0;
    elseif isstring(C) || ischar(C)
        cs = string(C);
        mask = cs ~= "background" & cs ~= "bg" & cs ~= "0";
        mask = mask(:,:,1);
    else
        mask = C(:,:,1) ~= 0;
    end
catch
    mask = false(size(C,1), size(C,2));
end
mask = logical(mask);
end
