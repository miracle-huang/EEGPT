import numpy as np
from scipy.integrate import simpson as simps
from scipy.fft import fft, ifft
from scipy.signal import butter, lfilter

def fourier_interpolation(signal, target_freq, original_freq):
    """
    使用傅里叶插值对信号进行上采样或下采样。
    signal: 输入的 1D 信号
    target_freq: 目标采样频率
    original_freq: 原始采样频率
    """
    n_samples = len(signal)  # 原始采样点数
    duration = n_samples / original_freq  # 信号持续时间
    target_samples = int(duration * target_freq)  # 目标采样点数

    # 傅里叶变换
    fft_result = fft(signal)

    # 调整频谱大小
    if target_samples > n_samples:
        # 上采样：在频域插入零
        pad_width = (target_samples - n_samples) // 2
        fft_result = np.pad(fft_result, (pad_width, pad_width), mode='constant')
    else:
        # 下采样：截断频域
        fft_result = fft_result[:target_samples]

    # 逆傅里叶变换
    upsampled_signal = ifft(fft_result).real
    return upsampled_signal

# 划分窗口
def segment_signal(signal, window_size, overlap):
    segments = []
    step = window_size - overlap
    for start in range(0, len(signal) - window_size + 1, step):
        end = start + window_size
        segments.append(signal[start:end])
    # return np.array(segments)  
    return segments

def calculate_0_and_nan(arr):
    '''计算数组中NaN和0的数量'''
    nan_count = np.isnan(arr).sum()
    non_nan_mask = ~np.isnan(arr)
    zero_count = (arr[non_nan_mask] == 0).sum()

    return nan_count, zero_count

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    对数据应用巴特沃斯带通滤波器
    
    参数:
    data -- 要过滤的输入信号数据
    lowcut -- 带通滤波器的低频截止频率(Hz)
    highcut -- 带通滤波器的高频截止频率(Hz)
    fs -- 采样频率(Hz)
    order -- 滤波器阶数，默认为5阶
    
    返回:
    y -- 经过带通滤波处理后的信号
    """
    
    # 计算奈奎斯特频率
    nyq = 0.5 * fs
    
    # 归一化截止频率
    low = lowcut / nyq
    high = highcut / nyq
    
    # 设计巴特沃斯带通滤波器
    b, a = butter(order, [low, high], btype='band')
    
    # 应用滤波器
    y = lfilter(b, a, data)
    
    return y