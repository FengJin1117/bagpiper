# 改成单卡，非分布式

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=100

# 分布式训练相关参数
# num_nodes=1       # 一个服务器
# num_proc_per_node=1 # 先单卡跑
# # num_proc_per_node=2 # 每个服务器上跑2个进程（一个进程一个GPU）
# node_rank=0       # 当前服务器的rank，单服务器训练时就是0
# master_addr=localhost   # 主节点地址，单服务器训练时就是localhost
# master_port=12346 # 主节点通信端口，确保这个端口在你的服务器上是空闲的

# 国内网络设置
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 网卡设置（训练时用）
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=WARN

# 两个基础任务配置

# ASR
# train_unregistered_specifier="audio_to_text:librispeech_train_960:manifest/train_960/dataset.json"

# 为了让它临时跑通
train_unregistered_specifier="audio_to_text:librispeech_test_clean:manifest/test_clean/dataset.json"
valid_unregistered_specifier="audio_to_text:librispeech_dev:manifest/dev/dataset.json"
test_unregistered_specifier="audio_to_text:librispeech_test_clean:manifest/test_clean/dataset.json"

# TTS
# train_unregistered_specifier="text_to_audio:librispeech_train_960:manifest/train_960/dataset.json"
# valid_unregistered_specifier="text_to_audio:librispeech_dev:manifest/dev/dataset.json"
# test_unregistered_specifier="text_to_audio:librispeech_test_clean:manifest/test_clean/dataset.json"

train_config=conf/train.yaml

stats_dir=exp/stats
# exp_dir=exp/librispeech_tts
exp_dir=exp/librispeech_asr

inference_config=conf/inference.yaml
inference_step=50000
inference_nj=1

. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

# 统计长度
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  python ../../../espnet2/speechlm/bin/prepare_length_stats.py \
    --train-unregistered-specifier "${train_unregistered_specifier}" \
    --valid-unregistered-specifier "${valid_unregistered_specifier}" \
    --train-config ${train_config} \
    --output-dir ${stats_dir} \
    --num-workers 4
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  deepspeed \
    ../../../espnet2/speechlm/bin/train_naive.py \
      --train-unregistered-specifier "${train_unregistered_specifier}" \
      --valid-unregistered-specifier "${valid_unregistered_specifier}" \
      --train-config ${train_config} \
      --stats-dir ${stats_dir} \
      --output-dir ${exp_dir} \
      --wandb-mode disabled

  # 纯python（不用deepspeed，会启动MPI，这里其实用不到）
  # python ../../../espnet2/speechlm/bin/train_naive.py \
  #     --train-unregistered-specifier "${train_unregistered_specifier}" \
  #     --valid-unregistered-specifier "${valid_unregistered_specifier}" \
  #     --train-config ${train_config} \
  #     --stats-dir ${stats_dir} \
  #     --output-dir ${exp_dir} \
  #     --wandb-mode disabled

fi


# 推理单独隔离出去了
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  inference_tag=$(basename "${inference_config%.*}")  # 这个命令什么意思？举例说明
  # 这个命令的作用是从inference_config的路径中提取出文件名（不带扩展名）作为inference_tag。
  # 举例说明：
  # 假设inference_config的值是 "conf/inference.yaml"，那么：
  # 1. "${inference_config%.*}" 会去掉文件扩展名，得到 "conf/inference"。
  # 2. basename "conf/inference" 会提取出最后的文件名部分，得到 "inference"。
  # 因此，inference_tag的值将会是 "inference"。


  inference_dir=${exp_dir}/inference/${inference_tag}_step_${inference_step}
  mkdir -p ${inference_dir}

  # checkpoint path
  inference_ckpt=${exp_dir}/checkpoints/step_${inference_step}/global_step${inference_step}/mp_rank_00_model_states.pt

  echo "Start model inference. Log at ${inference_dir}/logs/inference.*.log"
  ${cuda_cmd} JOB=1:${inference_nj} ${inference_dir}/logs/inference.JOB.log \
    ../../../espnet2/speechlm/bin/inference.py \
      --rank JOB --world-size ${inference_nj} \
      --train-config ${exp_dir}/train.yaml \
      --inference-config ${inference_config} \
      --model-checkpoint ${inference_ckpt} \
      --output-dir ${inference_dir} \
      --test-unregistered-specifier ${test_unregistered_specifier} \
      --num-worker 1
fi