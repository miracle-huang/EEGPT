import os
import math
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from Modules.Network.utils import Conv1dWithConstraint, LinearWithConstraint
import pytorch_lightning as pl
from functools import partial
import numpy as np
import random
from sklearn import metrics
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score, accuracy_score
from pytorch_lightning import Trainer
import time
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint, EarlyStopping

from utils import *
from utils_eval import get_metrics
from Modules.models.EEGPT_mcae import EEGTransformer

global max_epochs
global steps_per_epoch
global max_lr # 学习率调度器的最大学习率，在每次循环中设置为 8e-4

max_epochs = 30

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
seed_torch(42)

class LitEEGPTCausal(pl.LightningModule):
    def __init__(self, load_path="checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"):
        super().__init__()    
        # init model
        target_encoder = EEGTransformer(
            # 输入数据的大小，通常是一个二维张量的形状 [channels, time_points]
            # int(0.5*256)：表示时间维度的数据点数量，0.5 是时间长度（秒），256 是采样率（Hz）
            img_size=[32, int(0.5*256)], 
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

        # 通道标识符，用于指定要使用的通道，暂时写死为 DEAP 数据集的 32 个通道
        self.chans_id = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]])

        # -- load checkpoint
        pretrain_ckpt = torch.load(load_path)
        
        # 从预训练模型中提取目标编码器的参数（只加载预训练模型的target_encoder部分的权重参数）
        target_encoder_stat = {}
        for k,v in pretrain_ckpt['state_dict'].items():
            if k.startswith("target_encoder."):
                target_encoder_stat[k[15:]]=v

        # 定义了一个可学习的通道缩放参数，用于调整每个通道的重要性
        # 注意力机制：可以作为权重，用于对不同通道的特征赋予不同的重要性。
        self.chan_scale = torch.nn.Parameter(torch.ones(1, 32, 1)+ 0.001*torch.rand((1, 32, 1)), requires_grad=True)

        # 将预训练模型的参数加载到目标编码器中
        # load_state_dict方法将权重参数复制到模型中对应的层和参数中
        self.target_encoder.load_state_dict(target_encoder_stat)

        # 定义了两个线性层，用于将编码器提取的特征映射到最终的分类结果
        '''
        线性层的参数设置：
        2048 = 4 x 512 = embed_num x embed_dim
        48 = 3 x 16 = (时间窗口数) x (linear_probe1 的输出维度)
        '''
        self.linear_probe1   =   LinearWithConstraint(2048, 16, max_norm=1)
        self.linear_probe2   =   LinearWithConstraint(48, 2, max_norm=0.25)
       
        self.drop           = torch.nn.Dropout(p=0.50)
        
        self.loss_fn        = torch.nn.CrossEntropyLoss() # 定义了交叉熵损失函数
        
        self.running_scores = {"train":[], "valid":[], "test":[]}
        self.is_sanity = True

    def forward(self, x):
        # print(f"x.shape:{x.shape}") # B, C, T 
        '''
        B: Batch size (批次大小)
        C: Channels (通道数, 即EEG电极数量)
        T: Time steps (时间步长, 即采样点数量)
        '''
        B, C, T = x.shape

        x = x.to(torch.float) # 确保输入数据是浮点类型
        
        x = x - x.mean(dim=-2, keepdim=True) # 通道均值归一化,去除了共模噪声，突出了局部脑电活动

        x = x * self.chan_scale # 通过可学习的通道缩放参数调整每个通道的重要性
        # print(f"x.shape(after chan_scale):{x.shape}") # B, C, T 

        self.target_encoder.eval() 
        z = self.target_encoder(
            x, # 第一个参数：输入数据
            self.chans_id.to(x) # 第二个参数：通道标识符（已移至相同设备）
            ) 
        
        # print(f"z.shape:{z.shape}") # B, C, T 
        # print(f"x.shape(after process):{x.shape}") # B, C, T 
        
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
        self.running_scores["valid"]=[] #在每个验证周期开始时，创建一个空列表用于收集验证过程中的预测结果和真实标签
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
        
        # 计算多种评估指标
        metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "cohen_kappa", "f1", "roc_auc"]
        results = get_metrics(y_score.cpu().numpy(), label.cpu().numpy(), metrics, True)
        
        # 记录指标
        for key, value in results.items():
            self.log('valid_'+key, value, on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        '''
        定义模型在每个验证批次(batch)上的操作过程
        batch：一个批次的数据，通常由数据加载器提供，包含输入特征和目标标签。
        batch_idx：当前批次的索引，用于标识这是验证集中的第几个批次。
        '''
        x, y = batch # 解包 batch 为输入 x 和标签 y
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

    def test_step(self, batch, batch_idx):
        # 解包测试数据
        x, y = batch
        label = y.long()

        # 前向传播
        x, logit = self.forward(x)

        # 计算损失
        loss = self.loss_fn(logit, label)

        # 获取预测值和概率分布
        preds = torch.argmax(logit, dim=-1)
        y_score = torch.softmax(logit, dim=-1)[:, 1]  # 获取正类的概率

        # 计算准确率
        accuracy = ((preds==label)*1.0).mean()

        # 保存测试结果
        self.running_scores["test"].append((label.clone().detach().cpu(), preds.clone().detach().cpu(), y_score.clone().detach().cpu()))

        # 记录日志
        self.log('test_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)

        return loss

    def on_test_epoch_end(self) -> None:
        labels, preds, y_scores = [], [], []
        for label, pred, y_score in self.running_scores["test"]:
            labels.append(label)
            preds.append(pred)
            y_scores.append(y_score)

        # 将所有批次的数据拼接在一起
        labels = torch.cat(labels, dim=0).numpy()
        preds = torch.cat(preds, dim=0).numpy()
        y_scores = torch.cat(y_scores, dim=0).numpy()

        # 计算混淆矩阵
        conf_matrix = confusion_matrix(labels, preds)

        # 计算其他指标
        accuracy = accuracy_score(labels, preds)
        balanced_acc = balanced_accuracy_score(labels, preds)
        cohen_kappa = cohen_kappa_score(labels, preds)
        weighted_f1 = f1_score(labels, preds, average='weighted')
        auroc = roc_auc_score(labels, y_scores)

        # 打印混淆矩阵和指标
        print("Confusion Matrix:")
        print(conf_matrix)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Test Cohen Kappa: {cohen_kappa:.4f}")
        print(f"Test Weighted F1: {weighted_f1:.4f}")
        print(f"Test AUROC: {auroc:.4f}")

        # 记录日志
        self.log('test_acc', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_balanced_acc', balanced_acc, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_cohen_kappa', cohen_kappa, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_weighted_f1', weighted_f1, on_epoch=True, on_step=False, sync_dist=True)
        self.log('test_auroc', auroc, on_epoch=True, on_step=False, sync_dist=True)

        # 保存混淆矩阵和指标到文件
        os.makedirs("result", exist_ok=True)
        with open(f"result/test_results_video{video_id}.txt", "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(str(conf_matrix) + "\n\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Test Balanced Accuracy: {balanced_acc:.4f}\n")
            f.write(f"Test Cohen Kappa: {cohen_kappa:.4f}\n")
            f.write(f"Test Weighted F1: {weighted_f1:.4f}\n")
            f.write(f"Test AUROC: {auroc:.4f}\n")

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [self.chan_scale]+
            list(self.linear_probe1.parameters())+
            list(self.linear_probe2.parameters()),
            weight_decay=0.01)#
        
        # 是一种动态学习率调整策略，学习率会在一个周期内从较低值逐渐增加到最大值，然后再逐渐降低。
        # 这种策略能够在训练初期快速收敛，同时避免训练后期过早结束优化。
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

class MatDataset(Dataset):
    def __init__(self, mat_file_paths, data_key, label_key=None, split="train", transform=None):
        """
        初始化数据集
        :param mat_file_paths: .mat 文件路径列表
        :param data_key: 数据在 .mat 文件中的键名
        :param label_key: 标签在 .mat 文件中的键名（可选）
        :param split: 数据划分类型，"train", "val" 或 "test"
        :param transform: 数据变换（可选）
        """
        self.data = []
        self.labels = []
        self.transform = transform

        for mat_file_path in mat_file_paths:
            mat_data = scipy.io.loadmat(mat_file_path)
            data = torch.tensor(mat_data[data_key], dtype=torch.float32)
            labels = None
            if label_key:
                labels = torch.tensor(mat_data[label_key], dtype=torch.long).squeeze()

            # 根据 split 划分数据
            total_samples = len(data)
            train_end = int(0.8 * total_samples)  # 前 80% 的索引
            if split == "train":
                self.data.append(data[:train_end])
                if labels is not None:
                    self.labels.append(labels[:train_end])
            elif split == "val":
                self.data.append(data[train_end:])
                if labels is not None:
                    self.labels.append(labels[train_end:])
            elif split == "test":
                self.data.append(data)
                if labels is not None:
                    self.labels.append(labels)
            else:
                raise ValueError(f"Invalid split: {split}")

        # 合并来自多个文件的数据
        self.data = torch.cat(self.data, dim=0)
        if self.labels:
            self.labels = torch.cat(self.labels, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = None
        if self.labels is not None:
            label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return (sample, label) if label is not None else sample


if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()

    os.makedirs("result", exist_ok=True) # 创建结果文件夹

    # 假设所有 .mat 文件都存储在一个目录下
    mat_dir = "datasets/downstream/DEAP/video"
    mat_files = [os.path.join(mat_dir, f) for f in os.listdir(mat_dir) if f.endswith(".mat")]

    video_list = list(range(1, 41))
    for video_id in video_list:
        # 直接用EEGPT预训练模型进行迁移学习
        test_files = [f for f in mat_files if "video_" in f and int(f.split("_")[1].split(".")[0]) == video_id]   # 测试文件：编号等于 video_id
        print("现在用于测试的视频编号：", video_id)

        # 数据键名
        data_key = "data"  # 替换为 .mat 文件中的数据键名
        label_key = "valence_labels"  # 替换为 .mat 文件中的标签键名（如果有）

        # 创建测试集
        test_dataset = MatDataset(test_files, data_key, label_key, split="test")

        # 创建 DataLoader
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 验证数据加载
        print("Test dataset size:", len(test_dataset))

        # 加载预训练模型
        model = LitEEGPTCausal()

        # 创建测试日志记录器
        test_logger = pl_loggers.CSVLogger('./logs/', name="EEGPT_DEAP_video_test", version=f"video{video_id}")

        # 设置测试训练器
        test_trainer = pl.Trainer(
            accelerator='cuda',
            logger=[test_logger]
        )

        # 测试模型
        test_trainer.test(model, dataloaders=test_loader)
       
    # 记录结束时间
    end_time = time.time()

    # 计算运行时间
    execution_time = end_time - start_time
    print(f"代码运行时间: {execution_time:.6f} 秒")