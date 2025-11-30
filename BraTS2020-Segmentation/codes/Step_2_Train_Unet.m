%======================================================================
% 🎯 STEP 2 — Final Robust: Light Edition + Augmentation (Fixed)
%======================================================================
clear; clc; close all;
fprintf('Starting Step 2 (Final Robust Light Edition + Augmentation)\n');

%% -------------------- User config --------------------
data_root = 'BraTS2020_TrainingData';
image_dir = fullfile(data_root, 'Preprocessed_Data', 'Images');
mask_dir  = fullfile(data_root, 'Preprocessed_Data', 'Masks');

% How many pairs to use (adjust 200-2000 for speed vs quality)
numSubset = 1000;

% Resize target (must match U-Net input)
targetSize = [128 128];

% Network / training hyperparams for a light run
encoderDepth = 3;
numClasses   = 2;
initialLR    = 1e-4;
maxEpochs    = 10;
miniBatch    = 16;

%% -------------------- Validate paths & list files --------------------
if ~isfolder(image_dir) || ~isfolder(mask_dir)
    error('Preprocessed folders not found. Run Step 1 first.');
end

imageList = dir(fullfile(image_dir, '*.png'));
maskList  = dir(fullfile(mask_dir, '*.png'));

fprintf('Found %d image PNGs and %d mask PNGs.\n', numel(imageList), numel(maskList));
if numel(imageList) ~= numel(maskList)
    error('Image / mask count mismatch. Re-run Step 1 or check files.');
end

%% -------------------- Subset selection --------------------
numTotal = numel(imageList);
if numTotal > numSubset
    selIdx = randperm(numTotal, numSubset);
    imageList = imageList(selIdx);
    maskList  = maskList(selIdx);
    fprintf('Using a random subset: %d pairs (of %d total).\n', numel(imageList), numTotal);
else
    fprintf('Using all available pairs: %d.\n', numTotal);
end

% Build full file path cell arrays
imagePaths = fullfile({imageList.folder}, {imageList.name});
maskPaths  = fullfile({maskList.folder}, {maskList.name});

%% -------------------- Quick mask check --------------------
sampleMask = imread(maskPaths{1});
if islogical(sampleMask)
    sampleMask = uint8(sampleMask);
end
fprintf('Sample mask unique values: %s\n', mat2str(unique(sampleMask(:))'));

%% -------------------- Create datastores (plain) --------------------
imds = imageDatastore(imagePaths);
classNames = ["background","tumor"];
labelIDs   = [0 1];
pxds = pixelLabelDatastore(maskPaths, classNames, labelIDs);

%% -------------------- Train/Val split (indices) --------------------
numPairs = numel(imds.Files);
numTrain = round(0.8 * numPairs);
idxAll   = randperm(numPairs);
trainIdx = idxAll(1:numTrain);
valIdx   = idxAll(numTrain+1:end);

imdsTrain = subset(imds, trainIdx);
pxdsTrain = subset(pxds, trainIdx);
imdsVal   = subset(imds, valIdx);
pxdsVal   = subset(pxds, valIdx);

% Create pixelLabelImageDatastore with OutputSize (resizes on the fly)
plimdsTrain = pixelLabelImageDatastore(imdsTrain, pxdsTrain, 'OutputSize', targetSize);
plimdsVal   = pixelLabelImageDatastore(imdsVal, pxdsVal, 'OutputSize', targetSize);

% Save counts BEFORE applying transform (plimdsTrain will become TransformedDatastore)
trainCount = numel(plimdsTrain.ImageDatastore.Files);
valCount   = numel(plimdsVal.ImageDatastore.Files);
fprintf('Training pairs: %d, Validation pairs: %d\n', trainCount, valCount);

%% -------------------- Augmentation (applied only to training) --------------------
% Define augmentation transform function handle (robust to variable-name casing)
augmentFcn = @(data) augmentImageAndLabel_Robust(data, targetSize);

% Apply augmentation to training datastore (this returns a TransformedDatastore)
plimdsTrainAug = transform(plimdsTrain, augmentFcn);

%% -------------------- Define U-Net --------------------
inputSize = [targetSize 1];
lgraph = unetLayers(inputSize, numClasses, 'EncoderDepth', encoderDepth);

%% -------------------- Training options --------------------
opts = trainingOptions('adam', ...
    'InitialLearnRate', initialLR, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatch, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', plimdsVal, ...
    'ValidationFrequency', 50, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

fprintf('Starting training (this may take some time on CPU)...\n');

%% -------------------- Train network --------------------
% Use augmented training datastore
[net, info] = trainNetwork(plimdsTrainAug, lgraph, opts);

fprintf('✅ Training finished.\n');

%% -------------------- Save model --------------------
model_dir = fullfile(data_root, 'Trained_Unet_LightAug_Final');
if ~isfolder(model_dir)
    mkdir(model_dir);
end
save(fullfile(model_dir, 'trainedUnet_LightAug_final.mat'), 'net', 'info');
fprintf('💾 Model saved to: %s\n', model_dir);

%% -------------------- Quick visual check --------------------
try
    sample = read(plimdsVal);
    I = sample.InputImage{1};   % plimdsVal returns table with same var names as created
    L = sample.ResponsePixelLabelImage{1};
    C = semanticseg(I, net);
    B = labeloverlay(I, C, 'Transparency', 0.4);

    figure;
    subplot(1,3,1); imshow(I); title('Input (resized)');
    subplot(1,3,2); imshow(labeloverlay(I, L)); title('Ground Truth');
    subplot(1,3,3); imshow(B); title('Predicted');
    sgtitle('U-Net Validation Example');
catch ME
    warning('Could not produce quick visual check: %s', ME.message);
end

fprintf('✅ Step 2 (final, robust) complete.\n');

%% =================== Local helper functions ====================
function out = augmentImageAndLabel_Robust(data, imageSize)
    % Robust augmentation that discovers the input table's variable names
    % and returns a table using the same variable names.
    % data: table with one row and two cell columns (image + label) but the
    %       column names may vary in case (e.g., 'InputImage' or 'inputImage').

    vars = data.Properties.VariableNames;

    % Find likely image variable name
    imgVarCandidates = {'InputImage','inputImage','inputimage','inputimagefile','input'};
    imgVar = '';
    for k = 1:numel(imgVarCandidates)
        if any(strcmp(vars, imgVarCandidates{k}))
            imgVar = imgVarCandidates{k};
            break;
        end
    end
    if isempty(imgVar)
        % fallback: pick the first variable that contains 'image' (case-insensitive)
        idx = find(contains(lower(vars), 'image'), 1);
        if ~isempty(idx)
            imgVar = vars{idx};
        else
            imgVar = vars{1}; % extreme fallback
        end
    end

    % Find likely label variable name
    lblVarCandidates = {'ResponsePixelLabelImage','responsePixelLabelImage','responsepixellabelimage','Response','Label','Labels','response'};
    lblVar = '';
    for k = 1:numel(lblVarCandidates)
        if any(strcmp(vars, lblVarCandidates{k}))
            lblVar = lblVarCandidates{k};
            break;
        end
    end
    if isempty(lblVar)
        % fallback: pick the first variable that contains 'label' or 'response'
        idx = find(contains(lower(vars), 'label') | contains(lower(vars), 'response'), 1);
        if ~isempty(idx)
            lblVar = vars{idx};
        else
            % if nothing, pick second column if available
            if numel(vars) >= 2
                lblVar = vars{2};
            else
                lblVar = vars{1}; % extreme fallback
            end
        end
    end

    % Extract image and label using the discovered names
    I = data.(imgVar){1};
    L = data.(lblVar){1};

    % Ensure grayscale single-channel
    if ndims(I) == 3 && size(I,3) > 1
        I = rgb2gray(I);
    end

    % Resize (nearest for label)
    I = imresize(I, imageSize);
    L = imresize(L, imageSize, 'nearest');

    % Random horizontal flip
    if rand > 0.5
        I = fliplr(I); L = fliplr(L);
    end
    % Random vertical flip
    if rand > 0.5
        I = flipud(I); L = flipud(L);
    end
    % Random 90-degree rotation
    krot = randi([0,3]);
    if krot > 0
        I = rot90(I, krot);
        L = rot90(L, krot);
    end

    % Return table using the same original variable names (case preserved)
    out = table({I}, {L}, 'VariableNames', {imgVar, lblVar});
end
