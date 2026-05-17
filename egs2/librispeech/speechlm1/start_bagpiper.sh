# 启动模型FastAPI服务器，提供在线推理服务

set -e
set -u
set -o pipefail

# python ../../../espnet2/speechlm/bin/inference_single.py

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ========================
# 环境初始化（espnet必须）
# ========================
. ./path.sh
. ./cmd.sh

# 指定GPU
CUDA_VISIBLE_DEVICES=7 python ../../../espnet2/speechlm/inference_server.py