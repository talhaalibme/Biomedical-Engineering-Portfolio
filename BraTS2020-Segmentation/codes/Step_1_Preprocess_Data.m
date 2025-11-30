%% Step_1_Preprocess_Data.m
% Robust Step 1: Convert BraTS NIfTI (.nii / .nii.gz) volumes to PNG slices
% - Assumes data rooted at 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
% - Uses column 'BraTS_2020_subject_ID' from name_mapping.csv
% - Saves slices that contain tumor into Preprocessed_Data/Images and Masks
%
% Usage: run this file in MATLAB. Use TEST_MODE = true for a quick smoke test.

clear; clc; close all;
fprintf('Starting Step 1: Robust NIfTI -> PNG pre-processing\n');

%% ---------- CONFIG ----------
data_root = 'BraTS2020_TrainingData';
patient_data_subfolder = 'MICCAI_BraTS2020_TrainingData';
patient_data_path = fullfile(data_root, patient_data_subfolder);

mapping_csv  = fullfile(data_root, 'name_mapping.csv');
% outputs (placed inside data_root)
output_root = fullfile(data_root, 'Preprocessed_Data');
img_output_dir = fullfile(output_root, 'Images');
mask_output_dir = fullfile(output_root, 'Masks');

% Quick test mode (process only first TEST_N patients)
TEST_MODE = false;
TEST_N = 6;

% Make sure outputs exist
if ~exist(img_output_dir, 'dir'), mkdir(img_output_dir); end
if ~exist(mask_output_dir, 'dir'), mkdir(mask_output_dir); end

%% ---------- Basic checks ----------
if ~isfolder(patient_data_path)
    error('Patient data folder not found: %s\nPlease check data_root and patient_data_subfolder.', patient_data_path);
end
if ~isfile(mapping_csv)
    error('Mapping CSV not found: %s', mapping_csv);
end

fprintf('Patient data path: %s\n', patient_data_path);
fprintf('Mapping CSV: %s\n', mapping_csv);
fprintf('Output images: %s\n', img_output_dir);
fprintf('Output masks : %s\n\n', mask_output_dir);

%% ---------- Load mapping ----------
map_table = readtable(mapping_csv);

% Force the 2020 column (explicit)
patient_name_col = 'BraTS_2020_subject_ID';
if ~ismember(patient_name_col, map_table.Properties.VariableNames)
    error('Expected column "%s" not found in %s. Columns found: %s', patient_name_col, mapping_csv, strjoin(map_table.Properties.VariableNames, ', '));
end

% Extract and normalize patient id list into a cell array of chars
raw_ids = map_table.(patient_name_col);
if isstring(raw_ids), raw_ids = cellstr(raw_ids); end
if isnumeric(raw_ids), raw_ids = arrayfun(@num2str, raw_ids, 'UniformOutput', false); end
if iscell(raw_ids)
    patient_list = raw_ids;
else
    patient_list = cellstr(raw_ids);
end

% remove empty / NA entries
is_valid = ~cellfun(@(s) isempty(s) || any(strcmpi(strtrim(s), {'NA','NaN',''})), patient_list);
patient_list = patient_list(is_valid);

n_patients_total = numel(patient_list);
fprintf('Found %d patient IDs in mapping (after filtering empty/NA).\n\n', n_patients_total);

if TEST_MODE
    n_to_process = min(TEST_N, n_patients_total);
else
    n_to_process = n_patients_total;
end

%% ---------- Main loop ----------
patients_processed = 0;
slices_saved = 0;
tic;

for idx = 1:n_to_process
    patient_id = strtrim(patient_list{idx});
    if isempty(patient_id)
        continue;
    end

    % Try the straightforward folder first
    patient_folder = fullfile(patient_data_path, patient_id);

    % If not found, attempt alternative discovery strategies
    if ~isfolder(patient_folder)
        % 1) try wildcard under patient_data_path
        candidates = dir(fullfile(patient_data_path, [patient_id '*']));
        if ~isempty(candidates)
            % take first directory match
            dir_idx = find([candidates.isdir], 1);
            if ~isempty(dir_idx)
                patient_folder = fullfile(patient_data_path, candidates(dir_idx).name);
            else
                patient_folder = fullfile(patient_data_path, candidates(1).name);
            end
        else
            % 2) try matching numeric suffix
            nums = regexp(patient_id, '\d+', 'match');
            if ~isempty(nums)
                shortnum = nums{end};
                candidates2 = dir(fullfile(patient_data_path, ['*' shortnum '*']));
                if ~isempty(candidates2)
                    dir_idx = find([candidates2.isdir], 1);
                    if ~isempty(dir_idx)
                        patient_folder = fullfile(patient_data_path, candidates2(dir_idx).name);
                    else
                        patient_folder = fullfile(patient_data_path, candidates2(1).name);
                    end
                end
            end
        end
    end

    if ~isfolder(patient_folder)
        fprintf('WARNING: Patient folder not found for ID "%s". Tried: %s\n', patient_id, patient_folder);
        continue;
    end

    % Find .nii / .nii.gz files in patient folder
    nii_list = dir(fullfile(patient_folder, '*.nii*'));
    if isempty(nii_list)
        fprintf('WARNING: No .nii/.nii.gz files in folder %s. Skipping.\n', patient_folder);
        continue;
    end

    % Identify flair and seg files heuristically
    names = {nii_list.name};
    flair_idx = find(~cellfun(@isempty, regexpi(names, 'flair', 'once')), 1);
    seg_idx   = find(~cellfun(@isempty, regexpi(names, 'seg|segmentation|_gt|_label|mask', 'once')), 1);

    % fallback: explicit suffix patterns
    if isempty(flair_idx)
        flair_idx = find(~cellfun(@isempty, regexp(names, '_flair', 'once')), 1);
    end
    if isempty(seg_idx)
        seg_idx = find(~cellfun(@isempty, regexp(names, '_seg|_segmentation|_label', 'once')), 1);
    end

    % ultimate fallback: pick first and second file (best-effort)
    if isempty(flair_idx)
        flair_idx = 1;
    end
    if isempty(seg_idx)
        % if only one file exists assume seg is differently named -> skip to avoid bad results
        if numel(names) >= 2
            seg_idx = 2;
        else
            fprintf('WARNING: Could not identify segmentation file for %s, skipping.\n', patient_id);
            continue;
        end
    end

    flair_file = fullfile(patient_folder, names{flair_idx});
    seg_file   = fullfile(patient_folder, names{seg_idx});

    if mod(idx-1, 10) == 0
        fprintf('\nProcessing patient %s [%d / %d]...\n  flair: %s\n  seg  : %s\n', ...
            patient_id, idx, n_to_process, names{flair_idx}, names{seg_idx});
    end

    % Read volumes robustly
    try
        flair_vol = robust_nifti_read(flair_file);
        seg_vol   = robust_nifti_read(seg_file);
    catch ME
        fprintf('  WARNING: Failed reading NIfTI for %s: %s\n', patient_id, ME.message);
        continue;
    end

    % Normalize shapes: handle 4D volumes (channels last)
    if ndims(flair_vol) == 4 && size(flair_vol, 4) >= 1
        % take first channel (FLAIR usually channel 1 in some datasets)
        flair_vol = squeeze(flair_vol(:, :, :, 1));
    end
    if ndims(seg_vol) == 4 && size(seg_vol, 4) >= 1
        seg_vol = squeeze(seg_vol(:, :, :, 1));
    end

    % Validate sizes
    if any(size(flair_vol,1:3) ~= size(seg_vol,1:3))
        fprintf('  WARNING: Size mismatch (flair vs seg) for %s. Skipping.\n', patient_id);
        continue;
    end

    patients_processed = patients_processed + 1;

    % whole-tumor mask
    mask_vol_3D = (seg_vol > 0);

    n_slices = size(flair_vol, 3);
    for s = 1:n_slices
        img_slice = double(flair_vol(:, :, s));
        mask_slice = mask_vol_3D(:, :, s);

        if any(mask_slice(:))
            % Normalize to [0,1]
            maxv = max(img_slice(:));
            if maxv > 0
                img_norm = img_slice / maxv;
            else
                img_norm = img_slice;
            end

            fname = sprintf('%s_slice_%03d.png', patient_id, s);
            try
                imwrite(img_norm, fullfile(img_output_dir, fname));
                imwrite(mask_slice, fullfile(mask_output_dir, fname));
                slices_saved = slices_saved + 1;
            catch imE
                fprintf('    ERROR saving %s: %s\n', fname, imE.message);
            end
        end
    end
end

elapsed = toc;
fprintf('\n==================================================\n');
fprintf('Step 1 finished in %.2f seconds.\n', elapsed);
fprintf('  Processed %d patients.\n', patients_processed);
fprintf('  Saved %d PNG slices.\n', slices_saved);
fprintf('  Output folders:\n    %s\n    %s\n', img_output_dir, mask_output_dir);
fprintf('==================================================\n\n');

%% ===== Helper: robust NIfTI reader (must be at the end) =====
function vol = robust_nifti_read(fpath)
    % Try niftiread first, if not, handle .gz by gunzip, fallback to load_untouch_nii.
    try
        vol = niftiread(fpath);
        return;
    catch
        % continue to other strategies
    end

    % If compressed (.gz) -> gunzip then read
    if endsWith(lower(fpath), '.gz')
        tmpdir = tempname;
        mkdir(tmpdir);
        try
            gunzip(fpath, tmpdir);
            % find .nii in tmpdir
            d = dir(fullfile(tmpdir, '*.nii'));
            if isempty(d)
                error('After gunzip no .nii found in %s', tmpdir);
            end
            tmpnii = fullfile(tmpdir, d(1).name);
            vol = niftiread(tmpnii);
            % cleanup
            try rmdir(tmpdir, 's'); catch, end
            return;
        catch ME
            try rmdir(tmpdir, 's'); catch, end
            rethrow(ME);
        end
    end

    % Last resort: NIfTI toolbox function if present
    try
        nii = load_untouch_nii(fpath); %#ok<NASGU>
        vol = nii.img;
        return;
    catch
        error('robust_nifti_read: failed to read %s with available methods.', fpath);
    end
end
