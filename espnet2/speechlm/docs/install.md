# 关键文件

## bin/

1. inference.py：
- 多 GPU
- 大规模 dataset 推理


# 文件下载

hf download JinchuanTian/bagpiper_sft \
  --repo-type dataset \
  --include "exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_260000/*" \
  --local-dir step_260000

模型软链接：
ln -s /root/.cache/huggingface/hub/datasets--JinchuanTian--bagpiper_sft/snapshots/b11d5a0c11ad488edd04e3734d4bdff764977f57/exp ./exp

# 环境创建

```
conda create -n speechlm python=3.11 -y
conda activate speechlm

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118


<!-- pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -->

装flash-attn预编译：
torch2.5-cuda12-py311

# 升级 pip（最关键）
pip install --upgrade pip setuptools wheel

# 强制只用 wheel（避免一切编译）。用下载替代编译。
pip install -r requirement.txt --only-binary=:all:

```

后续的包补充
```
# 这是音频数据加载的需求：espnet2/speechlm/dataloader/multimodal_loader/audio_loader.py里用到的包，安装后就不会报错了。
pip install git+https://github.com/wanchichen/arkive.git
# 这个环境要走ssh
pip install git+ssh://git@github.com/wanchichen/arkive.git

# 后来证明，其实音频加载根本不用走这个，直接懒加载就行。
```

发现系统glibc版本过低，无法使用下载编译好的whl文件（要求glibc >= 2.32）
```
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```


pip命令升级：
pip install --upgrade pip


## 开始训练

1. protobuf版本过高
报错：`Downgrade the protobuf package to 3.20.x or lower.`

pip uninstall protobuf -y

pip install protobuf==3.20.3

2. deepspeed版本过高

```bash
pip uninstall deepspeed -y

pip install deepspeed==0.16.5
```

3. numpy版本过高，比wandb预期的

```
pip install numpy==1.26.4
python -c "import wandb; print('ok')"
```
# 对环境的理解

- 使用了比较新的qwen_2.5_omni，这是在老版本transformers库里没有的，所以需要安装最新版本的transformers。
这里也要求：transformers==4.57.1

- 而最新版本的transformers又要求：torch>=2.6.0
```
ValueError: Due to a serious vulnerability issue in torch.load, even with weights_only=True, we now require users to upgrade torch to 
**at least v2.6** in order to use the function. 
This version restriction does not apply when loading files with safetensors. See the vulnerability report here
```

```
# 卸载旧版本的torch
pip uninstall torch torchvision torchaudio -y

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118



