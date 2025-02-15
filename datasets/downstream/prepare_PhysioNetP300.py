import tqdm
import torch
import os
import mne

dataset_fold = "D:/huangzhiying/EEGPT/EEGPT/datasets/datasets/PhysioNetP300/finish_prepared_data"


all_chans = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

# fmin=0，fmax=120: 滤波器频率范围，用于滤波 EEG 信号，只保留 0Hz 到 120Hz 之间的频率分量
fmin=0
fmax=120
# tmin=-0.1，tmax=2: 时间窗口，指 EEG 事件相关电位（ERP） 的时间范围：
# -0.1s（事件前100ms） 到 2s（事件后2秒）。
tmin=-0.1
tmax=2
for sub in [2,3,4,5,6,7,9,11]:
    # path = "erp-based-brain-computer-interface-recordings-1.0.0/files/s{:02d}".format(sub)
    path = "D:/huangzhiying/EEGPT/EEGPT/datasets/datasets/PhysioNetP300/files/s{:02d}".format(sub)
    for file in os.listdir(path):
        if not file.endswith(".edf"):continue
        raw = mne.io.read_raw_edf(os.path.join(path, file)) # 读取edf文件
        raw.pick_channels(all_chans) # 仅保留预定义的 64 个 EEG 通道
        
        events, event_id = mne.events_from_annotations(raw) # 从标注中提取事件
        
        event_map = {}
        tgt = None # 记录 目标刺激（如 #Tgt1 中的 1）
        
        # 遍历 event_id，构建 event_map，用于将数字 事件 ID 映射回 原始标注（例如 1 → "#Tgt1"）
        for k,v in event_id.items():
            if k[0:4]=='#Tgt':
                tgt = k[4]
            event_map[v] = k
        # assert event_map[1][0:4]=='#Tgt' and event_map[2]=='#end' and event_map[3]=='#start', event_map
        assert tgt is not None # 确保目标刺激不为空

        '''
        mne.Epochs():
            以 tmin=-0.1s 到 tmax=2s 截取 EEG 片段。
            event_repeated='drop': 忽略重复的事件。
            preload=True: 预加载数据，加快处理速度。
        '''
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin = tmin, tmax=tmax,event_repeated='drop', preload=True, proj=False)#,event_repeated='drop',reject_by_annotation=True)
        
        '''
        先滤波再重采样
        根据奈奎斯特定理，重采样前应先滤波，以避免混叠。
        为了无失真地重建一个连续时间信号，采样频率 必须至少是信号 最高频率 的两倍
        '''
        epochs.filter(fmin, fmax,method = 'iir') # 滤波（0Hz - 120Hz）使用 IIR 滤波器
        epochs.resample(256)

        '''
        获取刺激事件 ID 和 EEG 数据。
        '''
        stims = [x[2] for x in epochs.events] # stims: 提取 epochs.events 中的 事件 ID。
        # print(stims)
        data = epochs.get_data() # data: 获取 epochs 的 EEG 数据，形状为 (样本数, 通道数, 时间步数)

        '''
        遍历所有 EEG 片段并保存
        '''
        for i,(d,t) in tqdm.tqdm(enumerate(zip(data, stims))):
            t = event_map[t] # 从 event_map 还原原始事件标注
            # 跳过 #Tgt, #end, #start 这些特殊标注
            if t.startswith('#Tgt') or t.startswith('#end') or t.startswith('#start') or t[0]=='#':
                continue
            # 若 tgt 在 t 中，则 label = 1（目标刺激）。否则 label = 0（非目标刺激）。
            label = 1 if tgt in t else 0
            # -- save
            x = torch.tensor(d*1e3)
            y = label
            spath = dataset_fold+f'/{y}/'
            directory = os.path.dirname(spath)
            print("dataset_fold:",dataset_fold)
            print("spath:",spath)
            if not os.path.exists(directory):
                os.makedirs(path,exist_ok=True)
            spath = spath + f'{i}.sub{sub}'
            torch.save(x, spath)
