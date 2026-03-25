clc;
clear;
close all;

% 加载数据
data = readtable('data.xlsx');

% 提取特征和分类标签
features = data(:, 1:9);
target = data(:, 10);

% 将特征和标签转换为数组
features = table2array(features);
target = table2array(target);

% 处理缺失值
features = rmmissing(features);

% 标准化特征
features = zscore(features);

% 使用t-SNE降维到二维
reducedFeatures = tsne(features);

% 创建一个新的表格，将降维后的特征和分类标签一起保存
outputTable = table(reducedFeatures(:,1), reducedFeatures(:,2), target, ...
    'VariableNames', {'Component1', 'Component2', 'Class'});

% 保存到Excel文件
writetable(outputTable, 'reduced_features.xlsx');

% 将每个分类标签单独保存为一列
uniqueClasses = unique(target);
for i = 1:length(uniqueClasses)
    classColumn = target == uniqueClasses(i);
    className = ['Class_' num2str(uniqueClasses(i))];
    outputTable.(className) = classColumn;
end

% 保存更新后的Excel文件，包含每个分类标签的独立列
writetable(outputTable, 'reduced_features_with_classes.xlsx');

% 绘制二维特征图
numClasses = length(uniqueClasses);
colors = lines(numClasses);

figure;
hold on;
for j = 1:numClasses
    % 按类别分散绘制数据点
    scatter(reducedFeatures(target == uniqueClasses(j), 1), reducedFeatures(target == uniqueClasses(j), 2), ...
            50, colors(j, :), 'filled', 'DisplayName', ['Class ', num2str(uniqueClasses(j))]);
end
hold off;
title('2D Feature Visualization with t-SNE');
xlabel('Component 1');
ylabel('Component 2');
legend('show');
