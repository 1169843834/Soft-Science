clc
clear all
close all
% 读取Excel文件中的数据
data = readtable('0.xlsx');

% 提取时间和样本数据
time = data{:, 1};
samples = data{:, 2:end};

% 获取采样频率和样本数
Fs = 50; % 采样频率为50Hz
N = length(time); % 样本数

% 设置窗口长度和重叠
window_length = 128; % 设置窗口长度（例如128个采样点）
overlap = window_length / 2; % 设置重叠部分（例如50%的重叠）
nfft = 256; % 设置FFT的点数

% 对每个通道进行短时傅里叶变换和绘图
for i = 1:size(samples, 2)
    % 获取当前通道数据
    channel_data = samples(:, i);
    
    % 计算短时傅里叶变换 (STFT)
    [S, f, t] = spectrogram(channel_data, window_length, overlap, nfft, Fs);
    
    % 创建新的Excel文件保存STFT的时频数据
    stft_data_filename = ['stft_results_channel_' num2str(i) '.xlsx'];
    
    % 设置图像窗口
    figure;
    
    % 绘制时频图（频率-时间图）
    surf(t, f, 10*log10(abs(S)), 'EdgeColor', 'none'); % 使用surf绘制3D图
    shading interp; % 平滑颜色填充
    axis xy; % 设置坐标轴方向
    title(['Channel ' num2str(i) ' Time-Frequency'], 'FontName', 'Times New Roman');
    xlabel('Time (s)', 'FontName', 'Times New Roman');
    ylabel('Frequency (Hz)', 'FontName', 'Times New Roman');
    set(gca, 'FontName', 'Times New Roman'); % 设置坐标轴字体
    colorbar; % 显示颜色条

    % 保存图像为JPEG格式
    saveas(gcf, ['stft_time_frequency_plot_channel_' num2str(i) '.jpeg']);
    
    % 将STFT后的数据写入Excel文件
    % 需要保证t和S数据的行列匹配，扩展t以使其与S的列数一致
    stft_table = table(repmat(t', length(f), 1), repmat(f, length(t), 1), reshape(abs(S), [], 1), ...
        'VariableNames', {'Time', 'Frequency', ['Magnitude_Channel_' num2str(i)]});
    
    % 将STFT表格数据写入Excel
    writetable(stft_table, stft_data_filename, 'WriteMode', 'overwrite');
    
    % 强制刷新图像窗口
    drawnow;
end

disp('STFT analysis complete. Results saved to individual files for each channel.');

