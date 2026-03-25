% 定义输入文件夹和输出文件夹路径
input_folder = 'C:\Users\11698\Desktop\NEWPAPER\Matlab\FFT'; % 请替换为实际的输入文件夹路径
output_folder = 'C:\Users\11698\Desktop\NEWPAPER\Matlab\FilteredFFT'; % 请替换为实际的输出文件夹路径

% 创建输出文件夹（如果不存在）
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% 文件名称列表
files = {'0.xlsx', '1.xlsx', '2.xlsx', '3.xlsx', '4.xlsx', '6.xlsx', '7.xlsx', '8.xlsx', '9.xlsx'};

% 设置采样频率和截止频率
Fs = 50; % 采样频率为50Hz
Fc_low = 3;  % 带通滤波器的下截止频率为3Hz

% 设计带通滤波器
Wn_low = Fc_low / (Fs / 2); % 归一化截止频率
[b, a] = butter(4, Wn_low, 'high'); % 4阶巴特沃斯高通滤波器（滤除小于3Hz的信号）

for k = 1:length(files)
    % 生成输入文件路径
    input_file = fullfile(input_folder, files{k});
    
    % 检查文件是否存在
    if ~isfile(input_file)
        fprintf('File not found: %s\n', input_file);
        continue;
    end
    
    % 读取数据
    data = readtable(input_file);

    % 提取时间和信号数据
    time = data{:, 1}; % 第一列为时间
    signals = data{:, 2:end}; % 第2到第10列为数据

    % 检查并处理无效值（Inf或NaN）
    signals(isinf(signals)) = NaN; % 将Inf值转换为NaN
    signals = fillmissing(signals, 'linear'); % 使用线性插值法填充NaN值

    % 再次检查是否存在无效值，若有则用0替换
    signals(~isfinite(signals)) = 0;

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
    sgtitle(['Original and Filtered Signals for file ', files{k}]);

    % 生成输出文件路径
    [~, file_name, ~] = fileparts(files{k});
    output_file = fullfile(output_folder, [file_name, '_filtered.xlsx']);
    
    % 保存滤波后的信号到新的Excel文件
    filtered_data = array2table([time, filtered_signals], ...
        'VariableNames', data.Properties.VariableNames);
    writetable(filtered_data, output_file);
end
