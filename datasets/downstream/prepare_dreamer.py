import numpy as np
import os
import scipy.io as sio

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# 获取当前脚本目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目主目录（config.py 所在位置）
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
# 添加项目主目录到 sys.path
sys.path.append(project_root)

from process_util import segment_signal, calculate_0_and_nan, fourier_interpolation, butter_bandpass_filter
import config

frequency = 128  # 采样频率
video_count = 18  # 视频数量
channel_count = 14  # 通道数量
subject_count = 23  # 受试者数量

def average_reference(video_eeg_stimuli):
    """
    对EEG数据进行平均参考处理
    """
    mean_ref = np.mean(video_eeg_stimuli, axis=1, keepdims=True)
    video_eeg_stimuli_ref = video_eeg_stimuli - mean_ref
    return video_eeg_stimuli_ref

def data_processing_dreamer():
    all_video_list_eeg = [
        [ [] for _ in range(channel_count) ]
        for _ in range(video_count)
    ]
    dreamer_dataset = sio.loadmat('datasets/datasets/Dreamer/DREAMER.mat')    
    dreamer_struct = dreamer_dataset['DREAMER']
    dreamer_data = dreamer_struct[0, 0]['Data']
    for subject_index in range(subject_count):
        print(f"Processing subject {subject_index + 1}/{subject_count}")
        subject_data = dreamer_data[0, subject_index]

        # 获取EEG数据
        eeg_data = subject_data['EEG'][0, 0]
        stimuli = eeg_data['stimuli'][0, 0]

        # 遍历每个视频
        for video_index in range(video_count):
            video_eeg_stimuli = stimuli[video_index][0]
            video_eeg_stimuli_ref = average_reference(video_eeg_stimuli) # dreamer没经过平均参考处理
            video_eeg = video_eeg_stimuli_ref.transpose((1, 0))  # 转置为 (channel, time)

            for channel_index in range(channel_count):
                channel_data_raw = video_eeg[channel_index]
                channel_data = butter_bandpass_filter(channel_data_raw, 0.5, 45, frequency, order=5) # dreamer没经过滤波

                nan_count, zero_count = calculate_0_and_nan(channel_data)
                if nan_count > 0:
                    print(f"视频 {video_index + 1} 通道 {config.dreamer_channel_mapping[channel_index]} 存在 {nan_count} 个 NaN 值")
                    all_video_list_eeg[video_index][channel_index].extend([])
                else:
                    upsampled_channel = fourier_interpolation(channel_data, target_freq=256, original_freq=128)
                    channel_windows = segment_signal(upsampled_channel, config.window_size_10, config.overlap)
                    # print(f"视频 {video_index + 1} 通道 {config.dreamer_channel_mapping[channel_index]} 窗口数量: {len(channel_windows)}")
                    all_video_list_eeg[video_index][channel_index].extend(channel_windows)

    print("所有视频处理完成，开始保存数据...")
    for video_index in range(video_count):
        video_eeg_array = np.array(all_video_list_eeg[video_index])
        video_eeg_array = np.transpose(video_eeg_array, (1, 0, 2))  # 转置为 (窗口数, 通道数, 窗口大小)
        arousal_labels_eeg = np.repeat(config.DREAMER_video_arousal_labels[int(video_index)], video_eeg_array.shape[0])
        valence_labels_eeg = np.repeat(config.DREAMER_video_valence_labels[int(video_index)], video_eeg_array.shape[0])

        processed_data_eeg = {
            'data': video_eeg_array,
            'arousal_labels': arousal_labels_eeg,
            'valence_labels': valence_labels_eeg
        }
        result_dir = 'datasets/downstream/dreamer/windowsize_10/'
        os.makedirs(result_dir, exist_ok=True)
        sio.savemat(result_dir + f"video_{video_index + 1}.mat", processed_data_eeg)
        print("Saved video {} processed video data".format(video_index + 1))

if __name__ == "__main__":
    data_processing_dreamer()