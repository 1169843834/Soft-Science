% 读取数据
% 读取数据
data = readtable('0.xlsx');

% 提取时间和信号数据
time = data{:, 1}; % 第一列为时间
signals = data{:, 2:end}; % 第2到第10列为数据

% 检查并处理无效值（Inf或NaN）
signals(isinf(signals)) = NaN; % 将Inf值转换为NaN
signals = fillmissing(signals, 'linear'); % 使用线性插值法填充NaN值

% 再次检查是否存在无效值，若有则用0替换
signals(~isfinite(signals)) = 0;

% 设置采样频率和截止频率
Fs = 50; % 采样频率为50Hz
Fc = 3;  % 截止频率为3Hz

% 设计低通滤波器
Wn = Fc / (Fs / 2); % 归一化截止频率
[b, a] = butter(4, Wn, 'low'); % 4阶巴特沃斯低通滤波器

% 对每个信号进行滤波
filtered_signals = filtfilt(b, a, signals);

% 绘制原始信号和滤波后的信号
figure;
for i = 1:size(signals, 2)
    subplot(size(signals, 2), 1, i);
    plot(time, signals(:, i), 'b'); hold on;
    plot(time, filtered_signals(:, i), 'r');
    xlabel('Time (s)');
    ylabel(['Signal ', num2str(i)]);
    legend('Original', 'Filtered');
end
sgtitle('Original and Filtered Signals');

% 保存滤波后的信号到新的Excel文件
filtered_data = array2table([time, filtered_signals], ...
    'VariableNames', data.Properties.VariableNames);
writetable(filtered_data, 'filtered_test.xlsx');
