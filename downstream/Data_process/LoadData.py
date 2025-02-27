import os
import glob
import numpy as np
import scipy.io as sio
import sys
import mne

'''
    This code is part of the fbcsptoolbox,which is a codebase in the github.
    Thanks for the https://fbcsptoolbox.github.io/ to provie this code.
'''

class LoadData:
    def __init__(self, eeg_file_path: str):
        self.eeg_file_path = eeg_file_path
        self.raw_eeg_subject = None

    def load_raw_data_gdf(self, file_to_load):
        # 加载 GDF 格式的 EEG（脑电）数据，并将其存储在 self.raw_eeg_subject 变量中。
        self.raw_eeg_subject = mne.io.read_raw_gdf(self.eeg_file_path + '/' + file_to_load)
        return self

    def load_raw_data_mat(self,file_to_load):
        self.raw_eeg_subject = sio.loadmat(self.eeg_file_path + '/' + file_to_load)

    def get_all_files(self, file_path_extension: str = None):
        if file_path_extension:
            return glob.glob(self.eeg_file_path+'/'+file_path_extension)
        return os.listdir(self.eeg_file_path)
    def filter(self,low,high):
        self.raw_eeg_subject.filter(low,high)

class LoadBCIC(LoadData):
    """Subclass of LoadData for loading BCI Competition IV Dataset 2a"""
    def __init__(self, file_to_load, *args):
        self.stimcodes = ['769', '770', '771', '772']
        # self.epoched_data={}
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        self.fs = None
        super(LoadBCIC, self).__init__(*args)

    def get_epochs(self, tmin=-4.5, tmax=5.0, bandpass = None,resample = None,baseline=None,reject = False):
        '''
        tmin	Epoch 开始时间（相对于事件触发时间），默认 -4.5 秒
        tmax	Epoch 结束时间，默认 5.0 秒
        bandpass	带通滤波的频率范围，例如 [1, 40] 代表 1-40Hz
        resample	目标采样率，若设定则进行重采样
        baseline	基线校正范围，如 (None, 0) 代表从 tmin 到 0s 进行校正
        reject	是否去除某些事件（如眨眼伪影），默认为 False
        '''

        # 加载 EEG 原始数据
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject # raw_data 现在存储了 MNE Raw 对象，即原始 EEG 数据

        # 提取事件信息
        # events：一个 (N, 3) 的数组，每行包含 [时间点, 0, 事件 ID]。
        # event_ids：一个字典，键是事件名称（字符串），值是对应的 ID（整数）
        events, event_ids = mne.events_from_annotations(raw_data)
        self.fs = raw_data.info.get('sfreq') # 获取采样频率

        # 如果 reject 为 True，则去除眨眼伪影事件
        if reject == True:
            reject_events = mne.pick_events(events,[1]) # 选取事件 ID 为 1 的事件
            reject_oneset = reject_events[:,0]/self.fs # 计算事件发生的时间点（秒）
            duration = [4]*len(reject_events) # 每个事件的持续时间 4s
            descriptions = ['bad trial']*len(reject_events) # 标记为“bad trial”
            blink_annot = mne.Annotations(reject_oneset,duration,descriptions) # 创建 MNE 注释
            raw_data.set_annotations(blink_annot) # 应用标注(将其标记为无效数据)
        
        # 提取 epochs
        # stims 选择需要的刺激事件（stimcodes 指定的事件）
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        '''
        tmin=-4.5, tmax=5.0：截取事件前 4.5s 到后 5s 的 EEG 数据。
        event_repeated='drop'：去除重复事件。
        baseline=baseline：基线校正（如 None, 0 代表使用 tmin 到 0s 作为基线）。
        preload=True：加载数据到内存，方便后续处理。
        reject_by_annotation=True：丢弃标记为 “bad trial” 的 epoch。
        '''
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=True)
        
        # 过滤 & 重采样
        if bandpass is not None:
            epochs.filter(bandpass[0],bandpass[1],method = 'iir') # 带通滤波（如 bandpass=[1, 40]）：保留 1Hz 到 40Hz 的 EEG 成分。
            # epochs.resample(128)
        if resample is not None:
            epochs.resample(resample) # 重采样（如 resample=128）：降低数据采样率，提高计算效率

        # 删除不需要的通道
        epochs = epochs.drop_channels(self.channels_to_remove)
        self.y_labels = epochs.events[:, -1] - min(epochs.events[:, -1]) # 取出所有 epoch 的事件 ID 作为标签, 使标签从 0 开始。
        self.x_data = epochs.get_data()*1e6 # 乘以 1e6，将数据转换为 μV
        # length = len(self.x_data)
        eeg_data={'x_data': self.x_data,
                  'y_labels': self.y_labels,
                  'fs': self.fs
                  }
        return eeg_data
    
class LoadBCIC_E(LoadData):
    """A class to lode the test data of the BICI IV 2a dataset"""
    def __init__(self, file_to_load, lable_name, *args):
        self.stimcodes = ('783')
        # self.epoched_data={}
        self.label_name = lable_name # the path of the test label
        self.file_to_load = file_to_load
        self.channels_to_remove = ['EOG-left', 'EOG-central', 'EOG-right']
        super(LoadBCIC_E, self).__init__(*args)

    def get_epochs(self, tmin=-4.5, tmax=5.0, bandpass = False,resample = None,baseline=None):
        self.load_raw_data_gdf(self.file_to_load)
        raw_data = self.raw_eeg_subject
        self.fs = raw_data.info.get('sfreq')
        events, event_ids = mne.events_from_annotations(raw_data)
        stims =[value for key, value in event_ids.items() if key in self.stimcodes]
        
        epochs = mne.Epochs(raw_data, events, event_id=stims, tmin=tmin, tmax=tmax, event_repeated='drop',
                            baseline=baseline, preload=True, proj=False, reject_by_annotation=False)



        if bandpass is not None:
            epochs.filter(bandpass[0],bandpass[1],method = 'iir')
            # epochs.resample(128)
        if resample is not None:
            epochs.resample(resample)
        epochs = epochs.drop_channels(self.channels_to_remove)
        label_info  = sio.loadmat(os.path.join(self.eeg_file_path,self.label_name))
        #label_info shape:(288, 1)
        self.y_labels = label_info['classlabel'].reshape(-1) -1
        # print(self.y_labels)
        self.x_data = epochs.get_data()*1e6
        eeg_data={'x_data': self.x_data,
                  'y_labels': self.y_labels,
                  'fs': self.fs}
        return eeg_data
    
class LoadBCIC_2b:
    '''A class to load the test and train data of the BICI IV 2b datast'''
    def __init__(self,path,subject,tmin =0,tmax = 4,bandpass = None):
        self.tmin = tmin
        self.tmax = tmax
        self.bandpass = bandpass
        self.subject = subject
        self.path = path
        self.train_name = ['1','2','3']
        self.test_name = ['4','5']
        self.train_stim_code  = ['769','770']
        self.test_stim_code  = ['783']
        self.channels_to_remove = ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        
    def get_train_data(self):
        data = []
        label = []
        for se in self.train_name:
            data_name = r'B0{}0{}T.gdf'.format(self.subject,se)
            label_name = r'B0{}0{}T.mat'.format(self.subject,se)
            data_path = os.path.join(self.path,data_name)
            label_path = os.path.join(self.path,label_name)
            data_x = self.get_epoch(data_path,True,self.tmin,self.tmax,self.bandpass)
            data_y = self.get_label(label_path)
            
            data.extend(data_x)
            label.extend(data_y)
        return np.array(data),np.array(label).reshape(-1)
    
    def get_test_data(self):
        data = []
        label = []
        for se in self.test_name:
            data_name = r'B0{}0{}E.gdf'.format(self.subject,se)
            label_name = r'B0{}0{}E.mat'.format(self.subject,se)
            data_path = os.path.join(self.path,data_name)
            label_path = os.path.join(self.path,label_name)
            data_x = self.get_epoch(data_path,False,self.tmin,self.tmax,self.bandpass)
            data_y = self.get_label(label_path)
            
            data.extend(data_x)
            label.extend(data_y)
        return np.array(data),np.array(label).reshape(-1)
            
    
    def get_epoch(self,data_path,isTrain = True,tmin =0,tmax = 4,bandpass = None):
        raw_data = mne.io.read_raw_gdf(data_path)
        events,events_id =  mne.events_from_annotations(raw_data)
        if isTrain:
            stims = [values for key,values in events_id.items() if key in self.train_stim_code]
        else:
            stims = [values for key,values in events_id.items() if key in self.test_stim_code]
        epochs = mne.Epochs(raw_data,events,stims,tmin = tmin,tmax = tmax,event_repeated='drop',baseline=None,preload=True, proj=False, reject_by_annotation=False)

        if bandpass is not None:
            epochs.filter(bandpass[0],bandpass[1],method = 'iir')

        epochs = epochs.drop_channels(self.channels_to_remove)
        eeg_data = epochs.get_data()*1e6
        return eeg_data
    
    def get_label(self,label_path):
        label_info = sio.loadmat(label_path)
        return label_info['classlabel'].reshape(-1)-1

class LoadHGD:
    '''
    A class to load the train data and test dats from raw data. 
    '''
    def __init__(self,path,subject) -> None:
        self.path = path
        self.subject = subject
        self.stims = [1,2,3,4]
        self.channel_names = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4',
                 'CP5', 'CP1', 'CP2', 'CP6',
                 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2', 'C6',
                 'CP3', 'CPz', 'CP4',
                 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h', 'FCC3h', 'FCC4h', 'FCC6h',
                 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h', 'CPP3h', 'CPP4h', 'CPP6h',
                 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h', 'CCP2h', 'CPP1h', 'CPP2h']
    
    def get_remove_channel(self,channel_list):
        channel_names =  ['EEG '+i for i in self.channel_names]
        return  [c for c in channel_list if c not in channel_names ]
         
    def get_epoch(self,file_path):
        raw_eeg = mne.io.read_raw_edf(file_path)
        events,event_id = mne.events_from_annotations(raw_eeg)
        drop_channels = self.get_remove_channel(raw_eeg.info['ch_names'])
        epoch = mne.Epochs(raw_eeg,events,event_id=self.stims,tmin = 0,tmax=4,event_repeated='drop',baseline=None,preload=True,proj=False,reject_by_annotation=True)
        epoch = epoch.drop_channels(drop_channels)
        # epoch.reject_by_annotation
        # epoch = epoch.filter(0.5,100)
        epoch = epoch.resample(250)
        
        x_data = epoch.get_data()
        y_data = epoch.events[:,-1]-min(epoch.events[:,-1])
        
        return x_data,y_data
    
    def get_train_data(self):
        train_path = os.path.join(self.path,'train','{}.edf'.format(self.subject))
        x_data,y_data = self.get_epoch(train_path)
        return np.array(x_data),np.array(y_data).reshape(-1)
    
    def get_test_data(self):
        test_path = os.path.join(self.path,'test','{}.edf'.format(self.subject))
        x_data,y_data = self.get_epoch(test_path)
        return np.array(x_data),np.array(y_data).reshape(-1)
 

 
class LoadKU(LoadData):
    """Subclass of LoadData for loading KU Dataset"""
    def __init__(self,subject_id,*args):
        self.subject_id=subject_id
        self.fs=1000
        super(LoadKU,self).__init__(*args)

    def get_epochs(self,sessions=[1, 2]):
        for i in sessions:
            file_to_load=f'session{str(i)}/s{str(self.subject_id)}/EEG_MI.mat'
            self.load_raw_data_mat(file_to_load)
            x_data = self.raw_eeg_subject['EEG_MI_train']['smt'][0, 0]
            x_data = np.transpose(x_data,axes=[1, 2, 0])
            labels = self.raw_eeg_subject['EEG_MI_train']['y_dec'][0, 0][0]
            y_labels = labels - np.min(labels)
            if hasattr(self, 'x_data'):
                self.x_data = np.append(self.x_data, x_data, axis=0)
                self.y_labels = np.append(self.y_labels, y_labels)
            else:
                self.x_data = x_data
                self.y_labels = y_labels
        ch_names = self.raw_eeg_subject['EEG_MI_train']['chan'][0, 0][0]
        ch_names_list = [str(x[0]) for x in ch_names]
        eeg_data = {'x_data': self.x_data,
                    'y_labels': self.y_labels,
                    'fs': self.fs,
                    'ch_names':ch_names_list}

        return eeg_data

if __name__ == '__main__':
    path = r'F:\pycharmProject\Git_project\EEG_Codebase\Raw_data\BCICIV_2a_gdf'
    data_path = r'/home/liuwenchao/code/code-test/Raw_data/BCICIV_2a_gdf/A01T.gdf'

    raw = mne.io.read_raw_gdf(data_path)

    info =  mne.io.read_info(raw)

    print(info['ch_names'])

    # load_raw_data = LoadBCIC_2b(path,5)
    # # train_x,train_y = load_raw_data.get_train_data()
    # test_x,test_y = load_raw_data.get_test_data()
    # print(test_x.shape,test_y.shape)
    # # print(train_x.shape,train_y.shape)