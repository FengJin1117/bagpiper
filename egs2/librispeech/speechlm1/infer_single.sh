#!/usr/bin/env bash

set -e
set -u
set -o pipefail

export HF_ENDPOINT=https://hf-mirror.com

# 这里避免每次启动，都网络请求来远端检查metadata。让模型直接用本地缓存，别磨叽！
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ckpt=exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_260000/global_step259988/mp_rank_00_model_states.pt

# train_config=conf/bagpiper/train.yaml
train_config=conf/train_flash.yaml
# train_config=conf/train_sdpa.yaml

inference_config=conf/inference.yaml

audio=examples/test.wav

. ./path.sh
. ./cmd.sh

CUDA_VISIBLE_DEVICES=5 python ../../../espnet2/speechlm/bin/inference_single.py \
    --audio ${audio} \
    --train-config ${train_config} \
    --inference-config ${inference_config} \
    --model-checkpoint ${ckpt}