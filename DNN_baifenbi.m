% 参数设置
dataSelectionRatio = 1; % 选择数据的比例
trainTestSplitRatio = 0.8; % 训练集和测试集的比例

% 读取数据
data = readmatrix('data.xlsx');
inputs = data(:, 1:9);
targets = data(:, 10);

% 将目标列转换为分类类型
targets = categorical(targets);

% 打乱数据集顺序
rng('default'); % 为了结果可重复性，设置随机种子
shuffledIdx = randperm(size(data, 1));
inputs = inputs(shuffledIdx, :);
targets = targets(shuffledIdx, :);

% 选择部分数据
numSelectedSamples = round(dataSelectionRatio * size(data, 1));
selectedInputs = inputs(1:numSelectedSamples, :);
selectedTargets = targets(1:numSelectedSamples, :);

% 划分选择的数据集为训练集和测试集
cv = cvpartition(numSelectedSamples, 'HoldOut', 1 - trainTestSplitRatio);
trainIdx = training(cv);
testIdx = test(cv);

trainInputs = selectedInputs(trainIdx, :);
trainTargets = selectedTargets(trainIdx, :);

testInputs = selectedInputs(testIdx, :);
testTargets = selectedTargets(testIdx, :);

% 设置层数和每层的神经元数量
numLayers = 3;
layerSizes = [50, 30, 20]; % 每层神经元数量

% 构建DNN网络架构
layers = [featureInputLayer(9)];
for i = 1:numLayers
    layers = [layers;
              fullyConnectedLayer(layerSizes(i));
              reluLayer];
end
layers = [layers;
          fullyConnectedLayer(17); % 输出层，9个分类
          softmaxLayer;
          classificationLayer];

% 设置训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {testInputs, testTargets}, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% 训练网络
net = trainNetwork(trainInputs, trainTargets, layers, options);

% 测试网络
predictions = classify(net, testInputs);

% 计算混淆矩阵
confMat = confusionmat(testTargets, predictions);
confMatDecimal = confMat ./ sum(confMat, 2); % 小数形式的混淆矩阵

% 计算召回率
recall = diag(confMat) ./ sum(confMat, 2);

% 计算整体召回率
overallRecall = mean(recall);

% 计算准确率
totalSamples = sum(confMat(:));
accuracy = sum(diag(confMat)) / totalSamples * 100; % 总准确率

% 计算每个分类的准确率
classAccuracy = diag(confMat) ./ sum(confMat, 1)'; % 每个分类的准确率
classAccuracy = classAccuracy * 100; % 转换为百分比

% 显示结果
disp('Confusion Matrix (Decimal):');
disp(confMatDecimal);
disp('Recall (each class):');
disp(recall);
disp(['Overall Recall: ', num2str(overallRecall)]);
disp(['Overall Accuracy: ', num2str(accuracy), '%']);
disp('Class Accuracy (each class):');
disp(classAccuracy);

% 绘制小数形式的混淆矩阵
figure;
confusionchart(testTargets, predictions, 'Normalization', 'row-normalized');
title('Confusion Matrix (Decimal)');

% 将混淆矩阵保存到Excel文件
confMatTable = array2table(confMatDecimal);
writetable(confMatTable, 'confusion_matrix.xlsx', 'WriteVariableNames', false);

% 保存召回率、整体召回率和准确率到Excel文件
resultsTable = table(recall, classAccuracy, 'VariableNames', {'Recall', 'ClassAccuracy'});
overallMetrics = table(overallRecall, accuracy, 'VariableNames', {'OverallRecall', 'OverallAccuracy'});
writetable(resultsTable, 'classification_results.xlsx', 'WriteRowNames', true);
writetable(overallMetrics, 'classification_results.xlsx', 'WriteMode', 'Append', 'WriteVariableNames', true);
