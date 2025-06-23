import os
import scipy.io as sio
from scipy.fft import fft, ifft
import numpy as np

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# 获取当前脚本目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目主目录（config.py 所在位置）
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# 添加项目主目录到 sys.path
sys.path.append(project_root)

# 数据集文件夹路径
dataset_folder = "datasets/datasets/DEAP/data_preprocessed_matlab/"

import config

# 指定需要保留的通道名称
use_channels_names = [
    'FP1', 'FPZ', 'FP2', 
    'AF3', 'AF4', 
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
    'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
    'O1', 'OZ', 'O2'
]

# DEAP 数据集中通道顺序（根据 DEAP 数据集文档）
deap_channel_names = [
    'FP1', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'OZ', 'PZ', 'FP2', 'AF4', 
    'FZ', 'F4', 'F8', 'FC6', 'T8', 'P8', 'O2', 'C3', 'C4', 'CZ', 'P3', 'P4', 
    'PO3', 'PO4', 'CP1', 'CP2', 'CPZ', 'FC1', 'FC2', 'F5'
]

# 创建通道名称到索引的映射
channel_mapping = {name: idx for idx, name in enumerate(deap_channel_names)}

# 获取需要保留的通道索引
use_channel_indices = [channel_mapping[name] for name in use_channels_names if name in channel_mapping]

def read_file(file):
    """
    读取单个 DEAP 数据集文件，提取指定通道的 EEG 数据，并上采样至 256Hz。
    """
    # 加载 .mat 文件
    data = sio.loadmat(file)
    file_name = file.split('/')[-1]

    # 提取 EEG 数据
    eeg_data = data['data'][:, :32, 384:]  # 去掉前 3 秒基线数据，维度为 (40, 32, 7680)

    # 提取标签
    labels = data['labels']  # 维度为 (40, 4)，分别为 valence, arousal, dominance, liking
    print(f"读取文件: {file_name}, EEG 数据维度: {eeg_data.shape}, 标签维度: {labels.shape}")

    # 上采样至 256Hz
    eeg_data_upsampled = []
    for trial in eeg_data:  # 遍历每个实验
        upsampled_trial = []
        for channel in trial:  # 遍历每个通道
            # 使用傅里叶变换插值
            upsampled_channel = fourier_interpolation(channel, target_freq=256, original_freq=128)
            upsampled_trial.append(upsampled_channel)
        eeg_data_upsampled.append(np.array(upsampled_trial))
    
    eeg_data_upsampled = np.array(eeg_data_upsampled)
    print(f"上采样后的 EEG 数据维度: {eeg_data_upsampled.shape}")
    
    # 过滤通道，只保留 use_channels_names 中的通道
    filtered_data = eeg_data_upsampled[:, use_channel_indices, :]
    print(f"过滤后的 EEG 数据维度: {filtered_data.shape}")
    return filtered_data, labels

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

def split_into_windows(data, labels, window_size=128):
    """
    将数据按照窗口大小分割，并分配标签。
    data: 输入数据，维度为 (trials, channels, samples)
    labels: 标签数据，维度为 (trials, 4)
    window_size: 每个窗口的采样点数
    """
    trials, channels, samples = data.shape
    num_windows = samples // window_size  # 每个 trial 中的窗口数量
    windows = []
    window_labels = []
    for trial in range(trials):
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            windows.append(data[trial, :, start:end])
            # 根据标签分配窗口标签           
            # valence_label = 1 if labels[trial, 0] > 5 else 0
            # arousal_label = 1 if labels[trial, 1] > 5 else 0
            # 根据视频平均得分排行设定窗口标签
            valence_label = 1 if trial + 1 in config.DEAP_half_valence_high else 0
            arousal_label = 1 if trial + 1 in config.DEAP_half_arousal_high else 0
            window_labels.append((arousal_label, valence_label))
    return np.array(windows), np.array(window_labels)

# def split_into_windows(data, labels, window_size=2560, step_size=256):
#     """
#     将数据按照窗口大小分割，并分配标签。
#     data: 输入数据，维度为 (trials, channels, samples)
#     labels: 标签数据，维度为 (trials, 4)
#     window_size: 每个窗口的采样点数
#     step_size: 滑动窗口的步长
#     """
#     trials, channels, samples = data.shape
#     windows = []
#     window_labels = []
#     for trial in range(trials):
#         for start in range(0, samples - window_size + 1, step_size):
#             end = start + window_size
#             windows.append(data[trial, :, start:end])
#             # 根据标签分配窗口标签           
#             valence_label = 1 if labels[trial, 0] > 5 else 0
#             arousal_label = 1 if labels[trial, 1] > 5 else 0
#             window_labels.append((arousal_label, valence_label))
#     return np.array(windows), np.array(window_labels)

def save_by_subject(data, labels, subject_id, output_folder="datasets/downstream/DEAP/window_size5/subject"):
    """
    按 subject 保存数据为 .mat 文件。
    data: 分割后的数据
    labels: 对应的标签
    subject_id: 被试编号
    output_folder: 保存路径
    """
    valence_labels = labels[:, 0]
    arousal_labels = labels[:, 1]  
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"subject_{subject_id}.mat")
    sio.savemat(output_path, {'data': data, 'valence_labels': valence_labels, 'arousal_labels': arousal_labels})
    print(f"按 subject 保存数据: {output_path}")

def save_by_video(data, labels, video_id, output_folder="datasets/downstream/DEAP/window_size5/video"):
    """
    按 video 保存数据为 .mat 文件。
    data: 分割后的数据
    labels: 对应的标签
    video_id: 视频编号
    output_folder: 保存路径
    """
    valence_labels = labels[:, 0]
    arousal_labels = labels[:, 1] 
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"video_{video_id}.mat")
    sio.savemat(output_path, {'data': data, 'valence_labels': valence_labels, 'arousal_labels': arousal_labels})
    print(f"按 video 保存数据: {output_path}")

if __name__ == '__main__':
    all_subject_data = []  # 用于存储每个 subject 的窗口化数据
    all_subject_labels = []  # 用于存储每个 subject 的窗口化标签

    # 遍历数据集文件夹，按 subject 读取和处理
    for file_name in os.listdir(dataset_folder):
        if file_name.endswith('.mat'):  # 只处理 .mat 文件
            subject_id = file_name.split('.')[0]  # 假设文件名为 s01.mat
            file_path = os.path.join(dataset_folder, file_name)
            # 读取文件并处理
            data, labels = read_file(file_path)  # 数据维度为 (trials, channels, samples)
            # 分割数据
            windows, window_labels = split_into_windows(data, labels, window_size=config.window_size_5)  # 窗口化数据维度为 (num_windows, channels, window_size)
            all_subject_data.append(windows)  # 按 subject 存储窗口化数据
            all_subject_labels.append(window_labels)  # 按 subject 存储窗口化标签
            # save_by_subject(windows, window_labels, subject_id)  # 按 subject 保存
            print(f"处理完成: {file_name}")
    
    # 按 video 保存
    num_trials = 40  # DEAP 数据集中每个 subject 有 40 个 trial
    num_windows_per_trial = all_subject_data[0].shape[0] // num_trials  # 每个 trial 的窗口数量
    for video_id in range(num_trials):  # 遍历每个 trial
        video_data = []
        video_labels = []
        for subject_id, subject_windows in enumerate(all_subject_data):
            # 计算当前 trial 的窗口索引范围
            start_idx = video_id * num_windows_per_trial
            end_idx = start_idx + num_windows_per_trial
            # 提取第 video_id 个 trial 的窗口化数据和标签
            video_data.append(subject_windows[start_idx:end_idx])  # 第 video_id 个 trial 的所有窗口
            video_labels.append(all_subject_labels[subject_id][start_idx:end_idx])  # 第 video_id 个 trial 的所有窗口标签
        # 合并所有被试的同一视频的窗口化数据和标签
        video_data = np.concatenate(video_data, axis=0)  # 将所有被试的窗口化数据合并
        video_labels = np.concatenate(video_labels, axis=0)  # 将所有被试的窗口化标签合并
        save_by_video(video_data, video_labels, video_id + 1)  # 保存按 video 的数据
        print(f"按 video 保存完成: video_{video_id + 1}")