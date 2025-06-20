frequency = 256  # 采样频率
window_size_10 = 2560  # 窗口大小为10秒
overlap = 0

DREAMER_video_valence_ratings = [3.17, 3.04, 4.57, 2.04, 3.22, 2.70, 4.52, 1.35, 1.39, 2.17, 3.96, 3.96, 4.39, 2.35, 2.48, 3.65, 1.52, 2.65]
DREAMER_video_arousal_ratings = [2.26, 3.00, 3.83, 4.26, 3.70, 3.83, 3.17, 3.96, 3.00, 3.30, 1.96, 2.61, 3.70, 2.22, 3.09, 3.35, 3.00, 3.91]
DREAMER_video_valence_labels = [1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
DREAMER_video_arousal_labels = [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
DREAMER_all_videos_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
DREAMER_half_valence_low = [4, 8, 9, 17]
DREAMER_half_valence_high = [3, 7, 12, 13]
DREAMER_half_arousal_low = [1, 11, 12, 14]
DREAMER_half_arousal_high = [3, 4, 8, 18]

AMIGO_video_valence_ratings = [6.99, 7.58, 3.74, 7.14, 3.88, 3.56, 3.46, 3.55, 3.3, 2.91, 3.15, 5.81, 6.75, 6.93, 7.64, 6.34]
AMIGO_video_arousal_ratings = [4.08, 4.23, 4.12, 3.84, 4.42, 5.15, 5.01, 6.79, 6.0, 5.59, 6.52, 5.64, 6.05, 4.38, 5.5, 5.53]
AMIGO_video_valence_labels = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
AMIGO_video_arousal_labels = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
AMIGO_all_videos_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
AMIGO_half_valence_low = [7, 9, 10, 11]
AMIGO_half_valence_high = [1, 2, 4, 15]
AMIGO_half_arousal_low = [1, 2, 3, 4]
AMIGO_half_arousal_high = [8, 9, 11, 13]

# channel names mapping
amigo_channel_mapping = {
    0: 'AF3',
    1: 'F7',
    2: 'F3',
    3: 'FC5',
    4: 'T7',
    5: 'P7',
    6: 'O1',
    7: 'O2',
    8: 'P8',
    9: 'T8',
    10: 'FC6',
    11: 'F4',
    12: 'F8',
    13: 'AF4'
}
dreamer_channel_mapping = {
    0: 'AF3',
    1: 'F7',
    2: 'F3',
    3: 'FC5',
    4: 'T7',
    5: 'P7',
    6: 'O1',
    7: 'O2',
    8: 'P8',
    9: 'T8',
    10: 'FC6',
    11: 'F4',
    12: 'F8',
    13: 'AF4'
}