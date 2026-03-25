%% run_dnn_compare_full_metrics.m
% MATLAB R2023a
% DNN baseline with:
% 5-fold CV + Accuracy + Macro-F1 + RRMSE + Train time
% + Inference time + Trainable parameters + Model size

clear; clc; close all;
rng(2026, "twister");

%% ===================== USER CONFIG =====================
cfg.fileName      = 'data1.xlsx';
cfg.signalCols    = 1:9;
cfg.labelCol      = 10;

cfg.kFolds        = 5;
cfg.maxEpochs     = 60;
cfg.miniBatchSize = 32;
cfg.initialLearnRate = 1e-3;

% 优化版 DNN 结构
cfg.layerSizes    = [64, 32, 16];

cfg.resultsRoot   = fullfile(pwd, 'results_full_metrics');
outDir = fullfile(cfg.resultsRoot, 'DNN');

if ~exist(cfg.resultsRoot, 'dir')
    mkdir(cfg.resultsRoot);
end
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% ===================== LOAD DATA =====================
T = readtable(cfg.fileName, 'VariableNamingRule','preserve');

if width(T) < 10
    error('The file must contain at least 10 columns.');
end

X = table2array(T(:, cfg.signalCols));
Y = table2array(T(:, cfg.labelCol));

validMask = all(~isnan(X), 2) & ~isnan(Y);
X = X(validMask, :);
Y = Y(validMask, :);

if isnumeric(Y)
    Y = categorical(Y);
elseif iscellstr(Y) || isstring(Y)
    Y = categorical(string(Y));
else
    Y = categorical(Y);
end

fprintf('Rows after cleaning: %d\n', size(X,1));
fprintf('Classes found: %s\n', strjoin(string(categories(Y)), ', '));

%% ===================== 5-FOLD CV =====================
cv = cvpartition(Y, 'KFold', cfg.kFolds);

foldAccuracy  = nan(cfg.kFolds,1);
foldMacroF1   = nan(cfg.kFolds,1);
foldRRMSE     = nan(cfg.kFolds,1);
foldTrainTime = nan(cfg.kFolds,1);

allTrue = categorical();
allPred = categorical();
foldDetail = table();

% 用最后一折训练好的网络统计部署指标
lastNet = [];
lastXTest = [];
lastYTest = [];

for fold = 1:cfg.kFolds
    fprintf('Fold %d/%d\n', fold, cfg.kFolds);

    trainIdx = training(cv, fold);
    testIdx  = test(cv, fold);

    XTrain = X(trainIdx, :);
    YTrain = Y(trainIdx);

    XTest  = X(testIdx, :);
    YTest  = Y(testIdx);

    % ===== training-data-only normalization =====
    mu = mean(XTrain, 1);
    sigma = std(XTrain, 0, 1);
    sigma(sigma < 1e-12) = 1;

    XTrain = (XTrain - mu) ./ sigma;
    XTest  = (XTest  - mu) ./ sigma;

    % ===== 训练集内部划分少量验证集，用于更稳定训练 =====
    cvInner = cvpartition(YTrain, 'HoldOut', 0.1);
    idxInnerTrain = training(cvInner);
    idxVal = test(cvInner);

    XTrainSub = XTrain(idxInnerTrain, :);
    YTrainSub = YTrain(idxInnerTrain);
    XVal      = XTrain(idxVal, :);
    YVal      = YTrain(idxVal);

    numFeatures = size(XTrainSub, 2);
    numClasses  = numel(categories(Y));

    layers = [
        featureInputLayer(numFeatures, 'Name','input', 'Normalization','none')

        fullyConnectedLayer(cfg.layerSizes(1), 'Name','fc1')
        batchNormalizationLayer('Name','bn1')
        reluLayer('Name','relu1')
        dropoutLayer(0.2, 'Name','drop1')

        fullyConnectedLayer(cfg.layerSizes(2), 'Name','fc2')
        batchNormalizationLayer('Name','bn2')
        reluLayer('Name','relu2')
        dropoutLayer(0.2, 'Name','drop2')

        fullyConnectedLayer(cfg.layerSizes(3), 'Name','fc3')
        reluLayer('Name','relu3')

        fullyConnectedLayer(numClasses, 'Name','fc_out')
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')
        ];

    options = trainingOptions('adam', ...
        'MaxEpochs', cfg.maxEpochs, ...
        'MiniBatchSize', cfg.miniBatchSize, ...
        'InitialLearnRate', cfg.initialLearnRate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 10, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {XVal, YVal}, ...
        'ValidationFrequency', 20, ...
        'ValidationPatience', 5, ...
        'Verbose', false, ...
        'Plots', 'none');

    % ===== Train =====
    tStart = tic;
    [net, info] = trainNetwork(XTrainSub, YTrainSub, layers, options);
    foldTrainTime(fold) = toc(tStart);

    % ===== Predict probabilities =====
    scores = predict(net, XTest, 'MiniBatchSize', cfg.miniBatchSize);

    classes = categories(Y);
    [~, idx] = max(scores, [], 2);
    YPred = categorical(classes(idx), classes);

    % ===== Metrics =====
    acc = mean(YPred == YTest);
    foldAccuracy(fold) = acc;

    macroF1 = computeMacroF1(YTest, YPred);
    foldMacroF1(fold) = macroF1;

    rrmse = computeRRMSE(YTest, scores);
    foldRRMSE(fold) = rrmse;

    allTrue = [allTrue; YTest];
    allPred = [allPred; YPred];

    % ===== Save per-fold prediction =====
    foldPredTbl = table((1:numel(YTest))', string(YTest), string(YPred), ...
        'VariableNames', {'IndexInFold','TrueLabel','PredLabel'});
    writetable(foldPredTbl, fullfile(outDir, sprintf('predictions_fold_%d.xlsx', fold)));

    % ===== Save training curve =====
    saveTrainingCurveDNN(info, outDir, fold);

    % ===== Fold detail =====
    tmp = table(repmat(fold, numel(YTest), 1), string(YTest), string(YPred), ...
        'VariableNames', {'Fold','TrueLabel','PredLabel'});
    foldDetail = [foldDetail; tmp];

    % 记录最后一折模型
    lastNet = net;
    lastXTest = XTest;
    lastYTest = YTest;
end

%% ===================== DEPLOYMENT / COMPLEXITY METRICS =====================
numParams = countTrainableParameters(lastNet);

modelFilePath = fullfile(outDir, 'trained_dnn_model.mat');
modelSizeMB = getModelFileSizeMB(lastNet, modelFilePath);

[singleInferMs, batchInferMs] = measureInferenceTimeDNN(lastNet, lastXTest, 100, min(32, size(lastXTest,1)));

%% ===================== SAVE ALL OUTPUTS =====================
writetable(foldDetail, fullfile(outDir, 'predictions_all_folds.xlsx'));

foldTbl = table((1:cfg.kFolds)', foldAccuracy*100, foldMacroF1*100, ...
    foldRRMSE, foldTrainTime, ...
    'VariableNames', {'Fold','Accuracy_percent','MacroF1_percent','RRMSE','TrainTime_sec'});
writetable(foldTbl, fullfile(outDir, 'fold_metrics.xlsx'));

results.meanAcc = mean(foldAccuracy, 'omitnan');
results.stdAcc  = std(foldAccuracy,  'omitnan');

results.meanMacroF1 = mean(foldMacroF1, 'omitnan');
results.stdMacroF1  = std(foldMacroF1,  'omitnan');

results.meanRRMSE = mean(foldRRMSE, 'omitnan');
results.stdRRMSE  = std(foldRRMSE,  'omitnan');

results.meanTrainTime = mean(foldTrainTime, 'omitnan');

summaryTbl = table(results.meanAcc*100, results.stdAcc*100, ...
    results.meanMacroF1*100, results.stdMacroF1*100, ...
    results.meanRRMSE, results.stdRRMSE, ...
    results.meanTrainTime, ...
    singleInferMs, batchInferMs, numParams, modelSizeMB, ...
    'VariableNames', {'Accuracy_mean_percent','Accuracy_std_percent', ...
    'MacroF1_mean_percent','MacroF1_std_percent', ...
    'RRMSE_mean','RRMSE_std','TrainTime_mean_sec', ...
    'InferenceTime_single_ms','InferenceTime_batch_ms', ...
    'TrainableParameters','ModelSize_MB'});
writetable(summaryTbl, fullfile(outDir, 'summary_metrics.xlsx'));

saveConfusionMatrixDNN(allTrue, allPred, outDir);

fig = figure('Color','w','Position',[100 100 680 420]);
bar(foldAccuracy*100);
xlabel('Fold');
ylabel('Accuracy (%)');
title('DNN - 5-fold Accuracy');
grid on;
ylim([0, 100]);
saveas(fig, fullfile(outDir, 'fold_accuracy.png'));
close(fig);

fig = figure('Color','w','Position',[100 100 680 420]);
bar(foldMacroF1*100);
xlabel('Fold');
ylabel('Macro F1 (%)');
title('DNN - 5-fold Macro F1');
grid on;
ylim([0, 100]);
saveas(fig, fullfile(outDir, 'fold_macroF1.png'));
close(fig);

fig = figure('Color','w','Position',[100 100 680 420]);
bar(foldRRMSE);
xlabel('Fold');
ylabel('RRMSE');
title('DNN - 5-fold RRMSE');
grid on;
saveas(fig, fullfile(outDir, 'fold_rrmse.png'));
close(fig);

disp('DNN done.');
disp(summaryTbl);

%% ===================== LOCAL FUNCTIONS =====================

function macroF1 = computeMacroF1(YTrue, YPred)
classes = categories(YTrue);
f1List = nan(numel(classes),1);

for i = 1:numel(classes)
    c = categorical(classes(i), classes);

    tp = sum(YTrue == c & YPred == c);
    fp = sum(YTrue ~= c & YPred == c);
    fn = sum(YTrue == c & YPred ~= c);

    precision = tp / max(tp + fp, 1);
    recall    = tp / max(tp + fn, 1);

    if precision + recall == 0
        f1 = 0;
    else
        f1 = 2 * precision * recall / (precision + recall);
    end
    f1List(i) = f1;
end

macroF1 = mean(f1List, 'omitnan');
end

function rrmse = computeRRMSE(YTrue, scores)
classes = categories(YTrue);
numClasses = numel(classes);
N = numel(YTrue);

Y_onehot = zeros(N, numClasses);

for i = 1:N
    classIdx = find(strcmp(classes, char(YTrue(i))));
    Y_onehot(i, classIdx) = 1;
end

rmse = sqrt(mean((scores - Y_onehot).^2, 'all'));
denom = sqrt(mean(Y_onehot.^2, 'all'));

if denom < 1e-12
    rrmse = NaN;
else
    rrmse = rmse / denom;
end
end

function saveTrainingCurveDNN(info, outDir, fold)
fig = figure('Color','w','Position',[120 120 900 360]);

subplot(1,2,1);
if isfield(info, 'TrainingLoss') && ~isempty(info.TrainingLoss)
    plot(info.TrainingLoss, 'LineWidth',1.5);
    hold on;
end
if isfield(info, 'ValidationLoss') && ~isempty(info.ValidationLoss) && any(~isnan(info.ValidationLoss))
    plot(info.ValidationLoss, 'LineWidth',1.5);
    legend('Train','Validation','Location','best');
else
    legend('Train','Location','best');
end
xlabel('Iteration');
ylabel('Loss');
title(sprintf('DNN - Fold %d Loss', fold));
grid on;

subplot(1,2,2);
hasTrainAcc = isfield(info, 'TrainingAccuracy') && ~isempty(info.TrainingAccuracy) && any(~isnan(info.TrainingAccuracy));
hasValAcc   = isfield(info, 'ValidationAccuracy') && ~isempty(info.ValidationAccuracy) && any(~isnan(info.ValidationAccuracy));

if hasTrainAcc
    plot(info.TrainingAccuracy, 'LineWidth',1.5);
    hold on;
end
if hasValAcc
    plot(info.ValidationAccuracy, 'LineWidth',1.5);
end

xlabel('Iteration');
ylabel('Accuracy (%)');
title(sprintf('DNN - Fold %d Accuracy', fold));
grid on;

if hasTrainAcc && hasValAcc
    legend('Train','Validation','Location','best');
elseif hasTrainAcc
    legend('Train','Location','best');
elseif hasValAcc
    legend('Validation','Location','best');
end

saveas(fig, fullfile(outDir, sprintf('training_curve_fold_%d.png', fold)));
close(fig);
end

function saveConfusionMatrixDNN(YTrue, YPred, outDir)
cats = categories(YTrue);
cm = confusionmat(YTrue, YPred, 'Order', categorical(cats));

cmTbl = array2table(cm, 'VariableNames', matlab.lang.makeValidName(cats), ...
    'RowNames', cellstr(cats));
writetable(cmTbl, fullfile(outDir, 'confusion_matrix_counts.xlsx'), ...
    'WriteRowNames', true);

cmNorm = cm ./ max(sum(cm,2), 1);
cmNormTbl = array2table(cmNorm, 'VariableNames', matlab.lang.makeValidName(cats), ...
    'RowNames', cellstr(cats));
writetable(cmNormTbl, fullfile(outDir, 'confusion_matrix_normalized.xlsx'), ...
    'WriteRowNames', true);

fig = figure('Color','w','Position',[120 120 700 620]);
cc = confusionchart(YTrue, YPred, ...
    'Title', 'DNN - Overall Confusion Matrix', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');
cc.FontName = 'Arial';
saveas(fig, fullfile(outDir, 'confusion_matrix.png'));
close(fig);
end

function numParams = countTrainableParameters(net)
numParams = 0;

if isa(net, 'SeriesNetwork') || isa(net, 'DAGNetwork')
    layers = net.Layers;
    for i = 1:numel(layers)
        layer = layers(i);

        if isprop(layer, 'Weights') && ~isempty(layer.Weights)
            numParams = numParams + numel(layer.Weights);
        end
        if isprop(layer, 'Bias') && ~isempty(layer.Bias)
            numParams = numParams + numel(layer.Bias);
        end
        if isprop(layer, 'Offset') && ~isempty(layer.Offset)
            numParams = numParams + numel(layer.Offset);
        end
        if isprop(layer, 'Scale') && ~isempty(layer.Scale)
            numParams = numParams + numel(layer.Scale);
        end
    end
else
    warning('Network type not fully supported for direct parameter counting.');
    numParams = NaN;
end
end

function modelSizeMB = getModelFileSizeMB(net, savePath)
save(savePath, 'net', '-v7.3');
fileInfo = dir(savePath);
modelSizeMB = fileInfo.bytes / 1024 / 1024;
end

function [singleTimeMs, batchTimeMs] = measureInferenceTimeDNN(net, XTest, nRepeat, batchSize)
if nargin < 3, nRepeat = 50; end
if nargin < 4, batchSize = min(32, size(XTest,1)); end

x1 = XTest(1,:);
tic;
for i = 1:nRepeat
    predict(net, x1);
end
singleTimeMs = toc / nRepeat * 1000;

xb = XTest(1:batchSize,:);
tic;
for i = 1:nRepeat
    predict(net, xb);
end
batchTimeMs = toc / nRepeat * 1000;
end