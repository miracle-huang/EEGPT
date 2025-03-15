import random 
import os
from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from torch import nn
import pytorch_lightning as pl

from functools import partial
import numpy as np
import random
import os 
import tqdm
from pytorch_lightning import loggers as pl_loggers
import torch.nn.functional as F
def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(7)

from utils import *
from Modules.models.EEGPT_mcae import EEGTransformer

from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
from sklearn import metrics
from utils_eval import get_metrics


ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
ch_names = [x.upper() for x in ch_names]

use_channels_names1 = [      'FP1', 'FPZ', 'FP2', 
                               'AF3', 'AF4', 
            'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
        'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
            'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
        'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
             'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
                      'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
                               'O1', 'OZ', 'O2', ]
use_channels_names = []
channels_index = []
for x in use_channels_names1:
    if x in ch_names:
        channels_index.append(ch_names.index(x))
        use_channels_names.append(x)
print(channels_index)

class LitEEGPTCausal(pl.LightningModule):

    def __init__(self, load_path="checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()    
        self.chans_num = len(use_channels_names) # 选取的EEG通道数量
        print(f"self.chans_num:{self.chans_num}") # self.chans_num:58, 有58个通道被选取
        # init model
        target_encoder = EEGTransformer(
            # 输入数据的大小，通常是一个二维张量的形状 [channels, time_points]
            # self.chans_num：表示 EEG 数据的通道数量（即电极数量）。
            # int(2.1*256)：表示时间维度的数据点数量，2.1 是时间长度（秒），256 是采样率（Hz）
            img_size=[self.chans_num, int(2.1*256)], # [58, 538]
            # patch_size 决定了如何将输入数据（img_size）沿时间维度分割成多个小块（patch）
            # num_patches = (time_points - patch_size) // patch_stride + 1 = 15
            patch_size=32*2, 
            patch_stride = 32,
            
            # embed_num：表示 Transformer 的嵌入层的数量
            # embed_dim：嵌入层的维度，表示每个补丁在嵌入空间中的表示大小。
            # 如果数据维度较高或模型需要更多的特征表达能力，可以增加此值。
            embed_num=4,
            embed_dim=512,
            depth=8, # 表示 Transformer 的深度，即 Transformer 中包含多少个 Encoder 层
            num_heads=8, # 表示 Transformer 中的多头注意力机制的头数, 如果嵌入维度较高，可以增加 num_heads
            # 在 Transformer 块中，MLP 层的隐藏层大小是 embed_dim * mlp_ratio
            mlp_ratio=4.0, # 表示 MLP 层的维度是嵌入维度的多少倍 
            drop_rate=0.0, # Dropout 的比例，用于防止过拟合, drop_rate=0.0 表示不使用 Dropout
            attn_drop_rate=0.0, # 注意力机制中的 Dropout 比例
            drop_path_rate=0.0, # Stochastic Depth（随机深度） 的丢弃比例
            init_std=0.02, # 初始化参数的标准差
            qkv_bias=True, # 是否在 QKV 层中使用偏置
            norm_layer=partial(nn.LayerNorm, eps=1e-6)) # 归一化层的类型，这里使用 LayerNorm
            
        self.target_encoder = target_encoder
        # self.predictor      = predictor
        self.chans_id       = target_encoder.prepare_chan_ids(use_channels_names)
        
        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path)
        
        # 从预训练模型中提取目标编码器的参数（只加载预训练模型的target_encoder部分的权重参数）
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v
        
        # 定义了一个可学习的通道缩放参数，用于调整每个通道的重要性
        # 注意力机制：可以作为权重，用于对不同通道的特征赋予不同的重要性。
        self.chan_scale = torch.nn.Parameter(torch.ones(1, self.chans_num, 1)+ 0.001*torch.rand((1, self.chans_num, 1)), requires_grad=True)
        
        # 将预训练模型的参数加载到目标编码器中
        # load_state_dict方法将权重参数复制到模型中对应的层和参数中
        self.target_encoder.load_state_dict(target_encoder_stat)
        
        # 定义了两个线性层，用于将编码器提取的特征映射到最终的分类结果
        '''
        线性层的参数设置：
        2048 = 4 x 512 = embed_num x embed_dim
        240 = 15 x 16 = (时间窗口数) x (linear_probe1 的输出维度)
        '''
        self.linear_probe1   =   LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2   =   LinearWithConstraint(240, 2, max_norm=0.25)
       
        self.drop           = torch.nn.Dropout(p=0.50)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss() # 定义了交叉熵损失函数
        
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity = True
        
    
    def forward(self, x):
        print(f"x.shape:{x.shape}") # B, C, T ([64, 64, 538])
        '''
        B: Batch size (批次大小)
        C: Channels (通道数, 即EEG电极数量)
        T: Time steps (时间步长, 即采样点数量)
        '''
        B, C, T = x.shape
        
        x = x.to(torch.float) # 确保输入数据是浮点类型
        
        x = x - x.mean(dim=-2, keepdim=True) # 通道均值归一化,去除了共模噪声，突出了局部脑电活动
        
        x = x[:,channels_index,:] # 选取指定的通道
        x = x * self.chan_scale # 通过可学习的通道缩放参数调整每个通道的重要性
        print(f"x.shape(after chan_scale):{x.shape}") # B, C, T ([64, 58, 538])
        
        # 使用预训练编码器提取特征 z: 表示编码器提取的特征
        # z 是一个特征向量或嵌入向量，它是通过非线性变换从原始输入空间映射到特征空间的结果：z = f(x, c)
        # 批量维度 [64]：处理的样本数量
        # 时间窗口维度 [15]：将原始信号分割成15个时间窗口
        # 嵌入数量维度 [4]：由 embed_num=4 直接决定，生成4个不同的嵌入向量
        # 嵌入维度 [512]：由 embed_dim=512 直接决定，每个嵌入向量的维度
        # f 表示编码器，x 表示输入数据，c 表示通道标识符 (chans_id)
        self.target_encoder.eval() 
        z = self.target_encoder(
            x, # 第一个参数：输入数据
            self.chans_id.to(x) # 第二个参数：通道标识符（已移至相同设备）
            ) 
        # 
        print(f"z.shape:{z.shape}") # B, C, T ([64, 15, 4, 512])
        print(f"x.shape(after process):{x.shape}") # B, C, T ([64, 58, 538])
        
        h = z.flatten(2) # 编码器输出z的第2维之后的所有维度展平
        
        h = self.linear_probe1(self.drop(h))
        
        h = h.flatten(1)
        
        h = self.linear_probe2(h)
        
        return x, h
    
    # 在训练过程中计算和记录模型性能指标
    def on_train_epoch_start(self) -> None:
        self.running_scores["train"]=[]
        return super().on_train_epoch_start()
    def on_train_epoch_end(self) -> None:
        label, y_score = [], []
        for x,y in self.running_scores["train"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        rocauc = metrics.roc_auc_score(label, y_score)
        self.log('train_rocauc', rocauc, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_train_epoch_end()
    
    '''
    PyTorch Lightning 框架的一个特殊方法，由框架内部自动调用，而不是在代码中直接调用
    trainer.fit() 方法会自动调用 training_step() 方法
    '''
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        # batch 是数据加载器提供的一批数据，通常包含输入特征和目标标签
        # 将 batch 解包为输入 x（EEG 信号数据）和标签 y
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x) # 前向传播，返回处理后的输入数据x和预测的模型输出logit
        loss = self.loss_fn(logit, label)
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        y_score =  logit
        y_score =  torch.softmax(y_score, dim=-1)[:,1] # 将 logits 通过 softmax 函数转换为概率分布
        self.running_scores["train"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        
        # Logging to TensorBoard by default
        self.log('train_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_avg', x.mean(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_max', x.max(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_min', x.min(), on_epoch=True, on_step=False, sync_dist=True)
        self.log('data_std', x.std(), on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"]=[]
        return super().on_validation_epoch_start()
    def on_validation_epoch_end(self) -> None:
        if self.is_sanity:
            self.is_sanity=False
            return super().on_validation_epoch_end()
            
        label, y_score = [], []
        for x,y in self.running_scores["valid"]:
            label.append(x)
            y_score.append(y)
        label = torch.cat(label, dim=0)
        y_score = torch.cat(y_score, dim=0)
        print(label.shape, y_score.shape)
        
        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, True)
        
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        label = y.long()
        
        x, logit = self.forward(x)
        
        preds = torch.argmax(logit, dim=-1)
        accuracy = ((preds==label)*1.0).mean()
        
        loss = self.loss_fn(logit, label)
        y_score =  logit
        y_score =  torch.softmax(y_score, dim=-1)[:,1]
        self.running_scores["valid"].append((label.clone().detach().cpu(), y_score.clone().detach().cpu()))
        # Logging to TensorBoard by default
        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(
            [self.chan_scale]+
            list(self.linear_probe1.parameters())+
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)#
        
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=max_epochs, pct_start=0.2)
        lr_dict = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            # The unit of the scheduler's step size, could also be 'step'
            'interval': 'step',
            'frequency': 1, # The frequency of the scheduler
            'monitor': 'val_loss', # Metric for `ReduceLROnPlateau` to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for `LearningRateMonitor` to use
        }
      
        return (
            {'optimizer': optimizer, 'lr_scheduler': lr_dict},
        )
        
# load configs
# -- LOSO 

# load configs
import torchvision
import math


global max_epochs
global steps_per_epoch
global max_lr

batch_size=64
max_epochs = 100

all_subjects = [1,2,3,4,5,6,7,9,11]
for i,sub in enumerate(all_subjects):
    sub_train = [f".sub{x}" for x in all_subjects if x!=sub]
    sub_valid = [f".sub{sub}"]
    print(sub_train, sub_valid)
    train_dataset = torchvision.datasets.DatasetFolder(root="datasets/downstream/PhysioNetP300", loader=torch.load, extensions=sub_train)
    valid_dataset = torchvision.datasets.DatasetFolder(root="datasets/downstream/PhysioNetP300", loader=torch.load, extensions=sub_valid)

    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    # 在windows下，为了避免出现死锁，num_workers=0
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    steps_per_epoch = math.ceil(len(train_loader))
    

    # init model
    model = LitEEGPTCausal()

    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor]
    max_lr = 8e-4
    trainer = pl.Trainer(accelerator='cuda',
                max_epochs=max_epochs, 
                callbacks=callbacks,
                enable_checkpointing=False,
                logger=[pl_loggers.TensorBoardLogger('./logs/', name="EEGPT_PhysioP300_tb", version=f"subject{sub}"), 
                        pl_loggers.CSVLogger('./logs/', name="EEGPT_PhysioP300_csv")])

    trainer.fit(model, train_loader, valid_loader, ckpt_path='last')