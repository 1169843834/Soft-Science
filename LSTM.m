%% run_lstm_compare_full_metrics.m
clear; clc; close all;
rng(2026, "twister");

%% ===================== USER CONFIG =====================
cfg.fileName      = 'data1.xlsx';
cfg.signalCols    = 1:9;
cfg.labelCol      = 10;

cfg.winLen        = 20;
cfg.winStride     = 5;
cfg.kFolds        = 5;
cfg.maxEpochs     = 40;
cfg.miniBatchSize = 32;
cfg.initialLearnRate = 1e-3;

cfg.resultsRoot   = fullfile(pwd, 'results_full_metrics');
outDir = fullfile(cfg.resultsRoot, 'LSTM');

if ~exist(cfg.resultsRoot, 'dir'); mkdir(cfg.resultsRoot); end
if ~exist(outDir, 'dir'); mkdir(outDir); end

%% ===================== LOAD DATA =====================
T = readtable(cfg.fileName, 'VariableNamingRule','preserve');

if width(T) < 10
    error('The file must contain at least 10 columns.');
end

Xraw = table2array(T(:, cfg.signalCols));
Yraw = table2array(T(:, cfg.labelCol));

validMask = all(~isnan(Xraw), 2) & ~isnan(Yraw);
Xraw = Xraw(validMask, :);
Yraw = Yraw(validMask, :);

if isnumeric(Yraw)
    Yraw = categorical(Yraw);
elseif iscellstr(Yraw) || isstring(Yraw)
    Yraw = categorical(string(Yraw));
else
    Yraw = categorical(Yraw);
end

[XSeq, YSeq] = buildSequenceWindows(Xraw, Yraw, cfg.winLen, cfg.winStride);

fprintf('LSTM windows: %d\n', numel(XSeq));

%% ===================== 5-FOLD CV =====================
cv = cvpartition(YSeq, 'KFold', cfg.kFolds);

foldAccuracy  = nan(cfg.kFolds,1);
foldMacroF1   = nan(cfg.kFolds,1);
foldRRMSE     = nan(cfg.kFolds,1);
foldTrainTime = nan(cfg.kFolds,1);

allTrue = categorical();
allPred = categorical();
foldDetail = table();

lastNet = [];
lastXTest = [];
lastYTest = [];

for fold = 1:cfg.kFolds
    fprintf('LSTM Fold %d/%d\n', fold, cfg.kFolds);

    trainIdx = training(cv, fold);
    testIdx  = test(cv, fold);

    XTrain = XSeq(trainIdx);
    YTrain = YSeq(trainIdx);
    XTest  = XSeq(testIdx);
    YTest  = YSeq(testIdx);

    [mu, sigma] = estimateSequenceMeanStd(XTrain);
    XTrain = normalizeSequenceCell(XTrain, mu, sigma);
    XTest  = normalizeSequenceCell(XTest,  mu, sigma);

    % 内部验证集
    cvInner = cvpartition(YTrain, 'HoldOut', 0.1);
    idxInnerTrain = training(cvInner);
    idxVal = test(cvInner);

    XTrainSub = XTrain(idxInnerTrain);
    YTrainSub = YTrain(idxInnerTrain);
    XVal      = XTrain(idxVal);
    YVal      = YTrain(idxVal);

    numFeatures = size(XTrainSub{1},1);
    numClasses  = numel(categories(YSeq));

    layers = [
        sequenceInputLayer(numFeatures, 'Normalization','none', 'Name','input')
        lstmLayer(64, 'OutputMode','last', 'Name','lstm')
        dropoutLayer(0.2, 'Name','drop1')
        fullyConnectedLayer(64, 'Name','fc1')
        reluLayer('Name','relu1')
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

    tStart = tic;
    [net, info] = trainNetwork(XTrainSub, YTrainSub, layers, options);
    foldTrainTime(fold) = toc(tStart);

    scores = predict(net, XTest, 'MiniBatchSize', cfg.miniBatchSize);

    classes = categories(YSeq);
    [~, idx] = max(scores, [], 2);
    YPred = categorical(classes(idx), classes);

    foldAccuracy(fold) = mean(YPred == YTest);
    foldMacroF1(fold)  = computeMacroF1(YTest, YPred);
    foldRRMSE(fold)    = computeRRMSE(YTest, scores);

    allTrue = [allTrue; YTest];
    allPred = [allPred; YPred];

    tmp = table(repmat(fold, numel(YTest), 1), string(YTest), string(YPred), ...
        'VariableNames', {'Fold','TrueLabel','PredLabel'});
    foldDetail = [foldDetail; tmp];

    saveTrainingCurve(info, outDir, fold, 'LSTM');

    lastNet = net;
    lastXTest = XTest;
    lastYTest = YTest;
end

%% ===================== DEPLOYMENT / COMPLEXITY METRICS =====================
numParams = countTrainableParameters(lastNet);

modelFilePath = fullfile(outDir, 'trained_lstm_model.mat');
modelSizeMB = getModelFileSizeMB(lastNet, modelFilePath);

[singleInferMs, batchInferMs] = measureInferenceTimeSeq(lastNet, lastXTest, 100, min(32, numel(lastXTest)), cfg.miniBatchSize);

%% ===================== SAVE ALL OUTPUTS =====================
saveAllOutputs(outDir, 'LSTM', cfg.kFolds, foldAccuracy, foldMacroF1, foldRRMSE, foldTrainTime, ...
    singleInferMs, batchInferMs, numParams, modelSizeMB, allTrue, allPred, foldDetail);

disp('LSTM done.');

%% ===================== LOCAL FUNCTIONS =====================
function [XSeq, YSeq] = buildSequenceWindows(Xraw, Yraw, winLen, stride)
XSeq = {};
YSeq = categorical();
cnt = 0;
for i = 1:stride:(size(Xraw,1)-winLen+1)
    ywin = Yraw(i:i+winLen-1);
    if numel(unique(ywin)) == 1
        cnt = cnt + 1;
        XSeq{cnt,1} = single(Xraw(i:i+winLen-1, :)');
        YSeq(cnt,1) = ywin(1);
    end
end
end

function [mu, sigma] = estimateSequenceMeanStd(XCell)
numFeatures = size(XCell{1},1);
sumX = zeros(numFeatures,1);
sumX2 = zeros(numFeatures,1);
N = 0;
for i = 1:numel(XCell)
    x = double(XCell{i});
    sumX = sumX + sum(x,2);
    sumX2 = sumX2 + sum(x.^2,2);
    N = N + size(x,2);
end
mu = sumX / N;
varx = sumX2 / N - mu.^2;
varx(varx < 1e-12) = 1e-12;
sigma = sqrt(varx);
end

function XCellN = normalizeSequenceCell(XCell, mu, sigma)
XCellN = XCell;
for i = 1:numel(XCell)
    x = double(XCell{i});
    x = (x - mu) ./ sigma;
    XCellN{i} = single(x);
end
end

function macroF1 = computeMacroF1(YTrue, YPred)
classes = categories(YTrue);
f1List = nan(numel(classes),1);
for i = 1:numel(classes)
    c = categorical(classes(i), classes);
    tp = sum(YTrue == c & YPred == c);
    fp = sum(YTrue ~= c & YPred == c);
    fn = sum(YTrue == c & YPred ~= c);
    precision = tp / max(tp + fp, 1);
    recall = tp / max(tp + fn, 1);
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
rrmse = rmse / denom;
end

function saveTrainingCurve(info, outDir, fold, modelName)
fig = figure('Color','w','Position',[120 120 900 360]);

subplot(1,2,1);
if isfield(info,'TrainingLoss') && ~isempty(info.TrainingLoss)
    plot(info.TrainingLoss,'LineWidth',1.5); hold on;
end
if isfield(info,'ValidationLoss') && ~isempty(info.ValidationLoss) && any(~isnan(info.ValidationLoss))
    plot(info.ValidationLoss,'LineWidth',1.5);
    legend('Train','Validation','Location','best');
else
    legend('Train','Location','best');
end
xlabel('Iteration'); ylabel('Loss');
title(sprintf('%s - Fold %d Loss', modelName, fold));
grid on;

subplot(1,2,2);
hasTrainAcc = isfield(info,'TrainingAccuracy') && ~isempty(info.TrainingAccuracy) && any(~isnan(info.TrainingAccuracy));
hasValAcc = isfield(info,'ValidationAccuracy') && ~isempty(info.ValidationAccuracy) && any(~isnan(info.ValidationAccuracy));
if hasTrainAcc
    plot(info.TrainingAccuracy,'LineWidth',1.5); hold on;
end
if hasValAcc
    plot(info.ValidationAccuracy,'LineWidth',1.5);
end
xlabel('Iteration'); ylabel('Accuracy (%)');
title(sprintf('%s - Fold %d Accuracy', modelName, fold));
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
        if isprop(layer, 'InputWeights') && ~isempty(layer.InputWeights)
            numParams = numParams + numel(layer.InputWeights);
        end
        if isprop(layer, 'RecurrentWeights') && ~isempty(layer.RecurrentWeights)
            numParams = numParams + numel(layer.RecurrentWeights);
        end
        if isprop(layer, 'Offset') && ~isempty(layer.Offset)
            numParams = numParams + numel(layer.Offset);
        end
        if isprop(layer, 'Scale') && ~isempty(layer.Scale)
            numParams = numParams + numel(layer.Scale);
        end
    end
else
    numParams = NaN;
end
end

function modelSizeMB = getModelFileSizeMB(net, savePath)
save(savePath, 'net', '-v7.3');
fileInfo = dir(savePath);
modelSizeMB = fileInfo.bytes / 1024 / 1024;
end

function [singleTimeMs, batchTimeMs] = measureInferenceTimeSeq(net, XTest, nRepeat, batchSize, miniBatchSize)
if nargin < 3, nRepeat = 50; end
if nargin < 4, batchSize = min(32, numel(XTest)); end
if nargin < 5, miniBatchSize = 32; end

x1 = XTest(1);
tic;
for i = 1:nRepeat
    predict(net, x1, 'MiniBatchSize', 1);
end
singleTimeMs = toc / nRepeat * 1000;

xb = XTest(1:batchSize);
tic;
for i = 1:nRepeat
    predict(net, xb, 'MiniBatchSize', miniBatchSize);
end
batchTimeMs = toc / nRepeat * 1000;
end

function saveAllOutputs(outDir, modelName, kFolds, foldAccuracy, foldMacroF1, foldRRMSE, foldTrainTime, ...
    singleInferMs, batchInferMs, numParams, modelSizeMB, allTrue, allPred, foldDetail)

writetable(foldDetail, fullfile(outDir, 'predictions_all_folds.xlsx'));

foldTbl = table((1:kFolds)', foldAccuracy*100, foldMacroF1*100, foldRRMSE, foldTrainTime, ...
    'VariableNames', {'Fold','Accuracy_percent','MacroF1_percent','RRMSE','TrainTime_sec'});
writetable(foldTbl, fullfile(outDir, 'fold_metrics.xlsx'));

summaryTbl = table(mean(foldAccuracy)*100, std(foldAccuracy)*100, ...
    mean(foldMacroF1)*100, std(foldMacroF1)*100, ...
    mean(foldRRMSE), std(foldRRMSE), mean(foldTrainTime), ...
    singleInferMs, batchInferMs, numParams, modelSizeMB, ...
    'VariableNames', {'Accuracy_mean_percent','Accuracy_std_percent', ...
    'MacroF1_mean_percent','MacroF1_std_percent', ...
    'RRMSE_mean','RRMSE_std','TrainTime_mean_sec', ...
    'InferenceTime_single_ms','InferenceTime_batch_ms', ...
    'TrainableParameters','ModelSize_MB'});
writetable(summaryTbl, fullfile(outDir, 'summary_metrics.xlsx'));

cats = categories(allTrue);
cm = confusionmat(allTrue, allPred, 'Order', categorical(cats));
cmTbl = array2table(cm, 'VariableNames', matlab.lang.makeValidName(cats), 'RowNames', cellstr(cats));
writetable(cmTbl, fullfile(outDir, 'confusion_matrix_counts.xlsx'), 'WriteRowNames', true);

cmNorm = cm ./ max(sum(cm,2),1);
cmNormTbl = array2table(cmNorm, 'VariableNames', matlab.lang.makeValidName(cats), 'RowNames', cellstr(cats));
writetable(cmNormTbl, fullfile(outDir, 'confusion_matrix_normalized.xlsx'), 'WriteRowNames', true);

fig = figure('Color','w','Position',[120 120 700 620]);
confusionchart(allTrue, allPred, 'Title', sprintf('%s - Overall Confusion Matrix', modelName), ...
    'RowSummary','row-normalized','ColumnSummary','column-normalized');
saveas(fig, fullfile(outDir, 'confusion_matrix.png'));
close(fig);

fig = figure('Color','w'); bar(foldAccuracy*100); xlabel('Fold'); ylabel('Accuracy (%)'); title([modelName ' - 5-fold Accuracy']); grid on;
saveas(fig, fullfile(outDir, 'fold_accuracy.png')); close(fig);

fig = figure('Color','w'); bar(foldMacroF1*100); xlabel('Fold'); ylabel('Macro F1 (%)'); title([modelName ' - 5-fold Macro F1']); grid on;
saveas(fig, fullfile(outDir, 'fold_macroF1.png')); close(fig);

fig = figure('Color','w'); bar(foldRRMSE); xlabel('Fold'); ylabel('RRMSE'); title([modelName ' - 5-fold RRMSE']); grid on;
saveas(fig, fullfile(outDir, 'fold_rrmse.png')); close(fig);
end