#!/usr/bin/env bash

set -e
set -u
set -o pipefail

# ========================
# 基本配置（你只需要改这里）
# ========================

export HF_ENDPOINT=https://hf-mirror.com


# ckpt（你给的路径）
# ckpt=exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_260000/global_step259988/mp_rank_00_model_states.pt
ckpt=exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_260000/universal_step259988/mp_rank_00_model_states.pt
# train config（必须，用来构建模型结构）
# train_config=conf/train.yaml
train_config=conf/bagpiper/train.yaml

# inference config
inference_config=conf/inference.yaml

# 数据（你可以改成自己的）
valid_unregistered_specifier="audio_to_text:librispeech_dev:manifest/dev/dataset.json"
test_unregistered_specifier="audio_to_text:librispeech_test_clean:manifest/test_clean/dataset.json"


# 输出目录
out_dir=exp/infer_custom

# 并行数（一般=1）
nj=1


# ========================
# 环境初始化（espnet必须）
# ========================
. ./path.sh
. ./cmd.sh


# ========================
# 开始推理
# ========================
mkdir -p ${out_dir}

echo "Start inference..."
echo "ckpt: ${ckpt}"

${cuda_cmd} JOB=1:${nj} ${out_dir}/log.JOB.txt \
    ../../../espnet2/speechlm/bin/inference.py \
        --rank JOB \
        --world-size ${nj} \
        --train-config ${train_config} \
        --inference-config ${inference_config} \
        --model-checkpoint ${ckpt} \
        --output-dir ${out_dir} \
        --test-unregistered-specifier "${test_unregistered_specifier}" \
        --num-worker 1

echo "Done."