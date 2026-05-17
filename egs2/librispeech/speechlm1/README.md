# 认知

这个 recipe 是针对UALM（更古早的模型）写的，比如train.yaml里的模型tag / size都是UALM的。不是Bagpiper



# 推理命令

```bash
bash prep.sh 
改成后台输出日志运行：
nohup bash prep.sh > prep.log 2>&1 &

# 开始训练
nohup bash infer.sh >logs/infer.log 2>&1 &

nohup bash infer_single.sh > logs/infer.log 2>&1 &
# 启动服务
nohup bash start_bagpiper.sh > logs/service.log 2>&1 &

```

# 训练命令


```bash
# 准备
CUDA_VISIBLE_DEVICES=5,6,7 nohup bash launch.sh --stage 1 --stop_stage 1 > logs/train_stage1.log 2>&1 &
# 开始训练
CUDA_VISIBLE_DEVICES=5,6,7 nohup bash launch.sh --stage 2 --stop_stage 2 > train_stage2.log 2>&1 &

# 单卡训练
CUDA_VISIBLE_DEVICES=5 nohup bash launch_naive.sh --stage 2 --stop_stage 2 > logs/train_naive_stage2.log 2>&1 &
```

注意：
deepspeed有两套GPU控制逻辑：CUDA_VISIBLE_DEVICES 和 num_nodes + num_proc_per_node参数互斥。


# 下载

```bash
# LLM
hf download Qwen/Qwen3-8B-Base

# 给我后台日志运行：
nohup bash download.sh > download.log 2>&1 &

# Audio encoder（最大）
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct

# codec
huggingface-cli download hf-audio/xcodec-hubert-general
```

# 问题：训练时OOM（单卡配置）

- 如果直接下载编译好的.whl，无法运行（gcc问题）