%% run_compare_4models_same_protocol.m
% MATLAB R2023a
% Compare DNN / 1D-CNN / LSTM / Ensemble under the same protocol

clear; clc; close all;
rng(2026, "twister");

%% ===================== USER CONFIG =====================
cfg.fileName      = 'data1.xlsx';
cfg.signalCols    = 1:9;
cfg.labelCol      = 10;

cfg.winLen        = 20;      % 20 time steps per sample
cfg.winStride     = 5;
cfg.kFolds        = 5;

cfg.maxEpochs     = 40;
cfg.miniBatchSize = 32;
cfg.initialLearnRate = 1e-3;

cfg.resultsRoot   = fullfile(pwd, 'results_4models_same_protocol');
if ~exist(cfg.resultsRoot, 'dir')
    mkdir(cfg.resultsRoot);
end

modelList = {'DNN','CNN','LSTM','Ensemble'};

%% ===================== LOAD DATA =====================
T = readtable(cfg.fileName, 'VariableNamingRule','preserve');

Xraw = table2array(T(:, cfg.signalCols));
Yraw = table2array(T(:, cfg.labelCol));

validMask = all(~isnan(Xraw),2) & ~isnan(Yraw);
Xraw = Xraw(validMask,:);
Yraw = Yraw(validMask,:);

if isnumeric(Yraw)
    Yraw = categorical(Yraw);
else
    Yraw = categorical(string(Yraw));
end

fprintf('Rows after cleaning: %d\n', size(Xraw,1));
fprintf('Classes: %s\n', strjoin(string(categories(Yraw)), ', '));

%% ===================== WINDOWING =====================
[XSeq, XFlat, YSeq] = buildWindowsForAllModels(Xraw, Yraw, cfg.winLen, cfg.winStride);

fprintf('Generated window samples: %d\n', numel(YSeq));

%% ===================== CV SPLIT =====================
cv = cvpartition(YSeq, 'KFold', cfg.kFolds);

summaryCell = cell(numel(modelList), 4);

for m = 1:numel(modelList)
    modelName = modelList{m};
    fprintf('\n==================== %s ====================\n', modelName);

    outDir = fullfile(cfg.resultsRoot, modelName);
    if ~exist(outDir, 'dir')
        mkdir(outDir);
    end

    results = runModelCV(modelName, XSeq, XFlat, YSeq, cv, cfg, outDir);

    summaryCell{m,1} = modelName;
    summaryCell{m,2} = results.meanAcc * 100;
    summaryCell{m,3} = results.meanMacroF1 * 100;
    summaryCell{m,4} = results.meanRRMSE;
end

summaryTbl = cell2table(summaryCell, ...
    'VariableNames', {'Model','Accuracy_mean_percent','MacroF1_mean_percent','RRMSE_mean'});

writetable(summaryTbl, fullfile(cfg.resultsRoot, 'summary_4models.xlsx'));
disp(summaryTbl);

%% ===================== LOCAL FUNCTIONS =====================

function [XSeq, XFlat, YSeq] = buildWindowsForAllModels(Xraw, Yraw, winLen, stride)
% XSeq : cell, each sample is [features x time]
% XFlat: numeric matrix, each sample is flattened [1 x (features*time)]

XSeq = {};
XFlat = [];
YSeq = categorical();

cnt = 0;
for i = 1:stride:(size(Xraw,1)-winLen+1)
    ywin = Yraw(i:i+winLen-1);
    if numel(unique(ywin)) == 1
        cnt = cnt + 1;
        seq = single(Xraw(i:i+winLen-1, :)');   % [9 x T]
        XSeq{cnt,1} = seq;
        XFlat(cnt,:) = reshape(seq, 1, []);     %#ok<AGROW>
        YSeq(cnt,1) = ywin(1);
    end
end
end

function results = runModelCV(modelName, XSeq, XFlat, YSeq, cv, cfg, outDir)

foldAccuracy = nan(cfg.kFolds,1);
foldMacroF1  = nan(cfg.kFolds,1);
foldRRMSE    = nan(cfg.kFolds,1);

allTrue = categorical();
allPred = categorical();

for fold = 1:cfg.kFolds
    fprintf('%s Fold %d/%d\n', modelName, fold, cfg.kFolds);

    trainIdx = training(cv, fold);
    testIdx  = test(cv, fold);

    YTrain = YSeq(trainIdx);
    YTest  = YSeq(testIdx);

    switch upper(modelName)
        case 'DNN'
            XTrain = XFlat(trainIdx,:);
            XTest  = XFlat(testIdx,:);

            [XTrain, XTest] = zscoreTrainTest(XTrain, XTest);
            [YPred, scores] = trainPredictDNN(XTrain, YTrain, XTest, cfg);

        case 'CNN'
            XTrain = XSeq(trainIdx);
            XTest  = XSeq(testIdx);

            [XTrain, XTest] = normalizeSeqTrainTest(XTrain, XTest);
            [YPred, scores] = trainPredictCNN(XTrain, YTrain, XTest, cfg);

        case 'LSTM'
            XTrain = XSeq(trainIdx);
            XTest  = XSeq(testIdx);

            [XTrain, XTest] = normalizeSeqTrainTest(XTrain, XTest);
            [YPred, scores] = trainPredictLSTM(XTrain, YTrain, XTest, cfg);

        case 'ENSEMBLE'
            XTrain = XFlat(trainIdx,:);
            XTest  = XFlat(testIdx,:);

            [XTrain, XTest] = zscoreTrainTest(XTrain, XTest);
            [YPred, scores] = trainPredictEnsemble(XTrain, YTrain, XTest);

        otherwise
            error('Unknown model: %s', modelName);
    end

    foldAccuracy(fold) = mean(YPred == YTest);
    foldMacroF1(fold)  = computeMacroF1(YTest, YPred);
    foldRRMSE(fold)    = computeRRMSE(YTest, scores);

    allTrue = [allTrue; YTest];
    allPred = [allPred; YPred];
end

results.meanAcc = mean(foldAccuracy,'omitnan');
results.meanMacroF1 = mean(foldMacroF1,'omitnan');
results.meanRRMSE = mean(foldRRMSE,'omitnan');

foldTbl = table((1:cfg.kFolds)', foldAccuracy*100, foldMacroF1*100, foldRRMSE, ...
    'VariableNames', {'Fold','Accuracy_percent','MacroF1_percent','RRMSE'});
writetable(foldTbl, fullfile(outDir, 'fold_metrics.xlsx'));

summaryTbl = table(results.meanAcc*100, std(foldAccuracy)*100, ...
    results.meanMacroF1*100, std(foldMacroF1)*100, ...
    results.meanRRMSE, std(foldRRMSE), ...
    'VariableNames', {'Accuracy_mean_percent','Accuracy_std_percent', ...
    'MacroF1_mean_percent','MacroF1_std_percent', ...
    'RRMSE_mean','RRMSE_std'});
writetable(summaryTbl, fullfile(outDir, 'summary_metrics.xlsx'));

fig = figure('Color','w','Position',[120 120 700 620]);
confusionchart(allTrue, allPred, 'RowSummary','row-normalized','ColumnSummary','column-normalized', ...
    'Title', [modelName ' - Overall Confusion Matrix']);
saveas(fig, fullfile(outDir, 'confusion_matrix.png'));
close(fig);
end

function [XTrainN, XTestN] = zscoreTrainTest(XTrain, XTest)
mu = mean(XTrain,1);
sigma = std(XTrain,0,1);
sigma(sigma<1e-12) = 1;
XTrainN = (XTrain - mu) ./ sigma;
XTestN  = (XTest - mu) ./ sigma;
end

function [XTrainN, XTestN] = normalizeSeqTrainTest(XTrain, XTest)
numFeatures = size(XTrain{1},1);

sumX = zeros(numFeatures,1);
sumX2 = zeros(numFeatures,1);
N = 0;
for i = 1:numel(XTrain)
    x = double(XTrain{i});
    sumX = sumX + sum(x,2);
    sumX2 = sumX2 + sum(x.^2,2);
    N = N + size(x,2);
end

mu = sumX / N;
varx = sumX2 / N - mu.^2;
varx(varx<1e-12) = 1e-12;
sigma = sqrt(varx);

XTrainN = XTrain;
XTestN = XTest;

for i = 1:numel(XTrain)
    XTrainN{i} = single((double(XTrain{i}) - mu) ./ sigma);
end
for i = 1:numel(XTest)
    XTestN{i} = single((double(XTest{i}) - mu) ./ sigma);
end
end

function [YPred, scores] = trainPredictDNN(XTrain, YTrain, XTest, cfg)
numFeatures = size(XTrain,2);
numClasses = numel(categories(YTrain));

layers = [
    featureInputLayer(numFeatures, 'Normalization','none')
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(16)
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', cfg.maxEpochs, ...
    'MiniBatchSize', cfg.miniBatchSize, ...
    'InitialLearnRate', cfg.initialLearnRate, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'none');

net = trainNetwork(XTrain, YTrain, layers, options);
scores = predict(net, XTest, 'MiniBatchSize', cfg.miniBatchSize);

classes = categories(YTrain);
[~, idx] = max(scores, [], 2);
YPred = categorical(classes(idx), classes);
end

function [YPred, scores] = trainPredictCNN(XTrain, YTrain, XTest, cfg)
numFeatures = size(XTrain{1},1);
numClasses = numel(categories(YTrain));

layers = [
    sequenceInputLayer(numFeatures, 'Normalization','none')
    convolution1dLayer(3, 32, 'Padding','same')
    reluLayer
    convolution1dLayer(3, 64, 'Padding','same')
    reluLayer
    globalAveragePooling1dLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', cfg.maxEpochs, ...
    'MiniBatchSize', cfg.miniBatchSize, ...
    'InitialLearnRate', cfg.initialLearnRate, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'none');

net = trainNetwork(XTrain, YTrain, layers, options);
scores = predict(net, XTest, 'MiniBatchSize', cfg.miniBatchSize);

classes = categories(YTrain);
[~, idx] = max(scores, [], 2);
YPred = categorical(classes(idx), classes);
end

function [YPred, scores] = trainPredictLSTM(XTrain, YTrain, XTest, cfg)
numFeatures = size(XTrain{1},1);
numClasses = numel(categories(YTrain));

layers = [
    sequenceInputLayer(numFeatures, 'Normalization','none')
    lstmLayer(64, 'OutputMode','last')
    dropoutLayer(0.2)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs', cfg.maxEpochs, ...
    'MiniBatchSize', cfg.miniBatchSize, ...
    'InitialLearnRate', cfg.initialLearnRate, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'none');

net = trainNetwork(XTrain, YTrain, layers, options);
scores = predict(net, XTest, 'MiniBatchSize', cfg.miniBatchSize);

classes = categories(YTrain);
[~, idx] = max(scores, [], 2);
YPred = categorical(classes(idx), classes);
end

function [YPred, scores] = trainPredictEnsemble(XTrain, YTrain, XTest)
% 这里用 LSBoost 树集成
mdl = fitcensemble(XTrain, YTrain, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 100, ...
    'Learners', templateTree('MaxNumSplits', 20));

[YPred, scoreTbl] = predict(mdl, XTest);

if istable(scoreTbl)
    scores = table2array(scoreTbl);
else
    scores = scoreTbl;
end

YPred = categorical(YPred);
end

function macroF1 = computeMacroF1(YTrue, YPred)
classes = categories(YTrue);
f1List = nan(numel(classes),1);

for i = 1:numel(classes)
    c = categorical(classes(i), classes);
    tp = sum(YTrue == c & YPred == c);
    fp = sum(YTrue ~= c & YPred == c);
    fn = sum(YTrue == c & YPred ~= c);

    precision = tp / max(tp+fp,1);
    recall = tp / max(tp+fn,1);

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