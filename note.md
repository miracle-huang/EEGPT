# TensorBoard使用方法
要读取日志文件并可视化，可以直接在命令行中启动 TensorBoard：
```
tensorboard --logdir ./logs/EEGPT_DEAP_video_tb
```
这个命令后面的路径是日记的保存路径
```
./logs/EEGPT_DEAP_video_tb/video{video_id}
```
- ./logs/ 是根目录。
- EEGPT_DEAP_video_tb 是子目录名称。
- video{video_id} 是指定版本的子目录（video_id 是动态变量）。
