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

from process_util import segment_signal, calculate_0_and_nan, fourier_interpolation
import config

dataset_dir = 'datasets/datasets/amigo/'
video_count = 16  # 视频数量
channel_count = 14  # 通道数量

def process_data_in_one_subject(signal_data):
    data_in_one_subject = []
    
    for video_index in range(video_count):
        video_data = [[] for _ in range(channel_count)]

        for channel_index in range(channel_count):
            channel_data = signal_data[video_index][:, channel_index]

            nan_count, zero_count = calculate_0_and_nan(channel_data)
            if nan_count > 0 or zero_count > 0:
                print(f"视频 {video_index + 1} 通道 {config.amigo_channel_mapping[channel_index]} 存在 {nan_count} 个 NaN 和 {zero_count} 个零值")
                video_data[channel_index].extend([])
            else:
                upsampled_channel = fourier_interpolation(channel_data, target_freq=256, original_freq=128)
                channel_windows = segment_signal(upsampled_channel, config.window_size_10, config.overlap)
                # print(f"视频 {video_index + 1} 通道 {config.amigo_channel_mapping[channel_index]} 窗口数量: {len(channel_windows)}")
                video_data[channel_index].extend(channel_windows)

        data_in_one_subject.append(video_data)

    return data_in_one_subject

def data_processing_amigo():
    all_video_list_eeg = [[[] for _ in range(channel_count)] for _ in range(video_count)]
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if filename.endswith('.mat'):
                mat_file_path = os.path.join(dirpath, filename)
                print(f"找到 .mat 文件: {mat_file_path}")

                all_data = sio.loadmat(mat_file_path)
                signal_data = all_data['joined_data'][0]
                data_in_one_subject = process_data_in_one_subject(signal_data)
                for video_index in range(video_count):
                    for channel_index in range(channel_count):
                        all_video_list_eeg[video_index][channel_index].extend(data_in_one_subject[video_index][channel_index])
    
    print("所有视频处理完成，开始保存数据...")
    for video_index in range(video_count):
        video_eeg_array = np.array(all_video_list_eeg[video_index])
        video_eeg_array = np.transpose(video_eeg_array, (1, 0, 2))  # 转置为 (窗口数, 通道数, 窗口大小)
        arousal_labels_eeg = np.repeat(config.AMIGO_video_arousal_labels[int(video_index)], video_eeg_array.shape[0])
        valence_labels_eeg = np.repeat(config.AMIGO_video_valence_labels[int(video_index)], video_eeg_array.shape[0])
        
        processed_data_eeg = {
            'data': video_eeg_array,
            'arousal_labels': arousal_labels_eeg,
            'valence_labels': valence_labels_eeg
        }
        result_dir = 'datasets/downstream/amigo/windowsize_10/'
        os.makedirs(result_dir, exist_ok=True)
        sio.savemat(result_dir + f"video_{video_index + 1}.mat", processed_data_eeg)
        print("Saved video {} processed video data".format(video_index + 1))

if __name__ == "__main__":
    data_processing_amigo()