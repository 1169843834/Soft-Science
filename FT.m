clc
clear all
close all
% 读取Excel文件中的数据
data = readtable('01.xlsx');

% 提取时间和样本数据
time = data{:, 1};
samples = data{:, 2:end};

% 获取采样频率和样本数
Fs = 50; % 采样频率为50Hz
N = length(time); % 样本数

% 频率轴
f = (0:(N/2))*(Fs/N);

% 初始化存储FFT和PSD结果的矩阵
fft_results = zeros(N/2+1, size(samples, 2));
psd_results = zeros(N/2+1, size(samples, 2));

% 创建新的Excel文件保存FFT和PSD结果
results_filename = 'results.xlsx';

% 设置图像窗口
figure;

% 对每个通道进行FFT和PSD计算
for i = 1:size(samples, 2)
    % 获取当前通道数据
    channel_data = samples(:, i);
    
    % 快速傅里叶变换 (FFT)
    Y = fft(channel_data);
    P2 = abs(Y/N);
    P1 = P2(1:N/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    
    % 存储FFT结果
    fft_results(:, i) = P1;
    
    % 绘制频率-幅值图
    subplot(2, size(samples, 2), i);
    plot(f, P1);
    title(['Channel ' num2str(i) ' Frequency-Amplitude']);
    xlabel('Frequency (Hz)');
    ylabel('Amplitude');
    
    % 计算能量谱密度 (PSD)
    psd = (1/(Fs*N)) * abs(Y).^2;
    psd(2:end-1) = 2*psd(2:end-1);
    
    % 存储PSD结果
    psd_results(:, i) = psd(1:N/2+1);
    
    % 绘制能量谱密度图
    subplot(2, size(samples, 2), i + size(samples, 2));
    plot(f, psd(1:N/2+1));
    title(['Channel ' num2str(i) ' Power Spectral Density']);
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
end

% 保存图像
saveas(gcf, 'fft_psd_plots.png');

% 将FFT结果写入Excel文件
fft_table = array2table([f', fft_results], 'VariableNames', ['Frequency', arrayfun(@(x) ['Channel_' num2str(x)], 1:size(samples, 2), 'UniformOutput', false)]);
writetable(fft_table, results_filename, 'Sheet', 'FFT');

% 将PSD结果写入Excel文件
psd_table = array2table([f', psd_results], 'VariableNames', ['Frequency', arrayfun(@(x) ['Channel_' num2str(x)], 1:size(samples, 2), 'UniformOutput', false)]);
writetable(psd_table, results_filename, 'Sheet', 'PSD');

disp('FFT and PSD analysis complete. Results saved to results.xlsx and fft_psd_plots.png.');
