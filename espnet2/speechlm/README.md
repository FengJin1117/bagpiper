# Bagpiper

# 新添：FastAPI封装

- 😎模型持久化：常驻GPU，避免重复下载，方便调试！

启动服务命令：
```bash
CUDA_VISIBLE_DEVICES=7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python bin/inference_server.py
```
> 实现细节：在service/文件夹下。

# 一些信息：

- global: 推理
- universal: 训练

两者都是bf16

TODO: 纯净版。其实recipe之间互不影响，
