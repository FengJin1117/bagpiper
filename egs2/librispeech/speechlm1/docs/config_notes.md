# inference.yaml解读

- 核心：分模态生成，生成text和audio使用不同的生成配置。

Bagpiper其实在做两件事：

- 生成 rich caption / CoT / answer（text）
- 生成 audio codec tokens（audio）

👉 两者本质完全不同，所以参数必须分开

```yaml
# LLM 推理常用“半精度”。生成模型如果用的是扩散模型（比如stable diffusion、DiT），一般不能用半精度，容易出数值不稳定问题（梯度爆炸/NaN）。但bagpiper是输出的token。
dtype: bfloat16
num_hypo: 1

# audio生成更多样
audio:
  temperature: 0.8
  topk: 20
  # cfg越大，越遵循prompt（这里其实是rich caption）。要求音频强对齐caption。
  cfg: 3
  max_step: 1024

text:
    # greedy decoding
    temperature: 0.0
    # 这里topk其实没啥用了
    topk: 20
    cfg: 1
    # 其实就是max_new_tokens
    max_step: 1024

```

# train.yaml （Bagpiper）

```yaml
job_type: speechlm

multimodal_io:
    text:
        tokenizer_name: Qwen/Qwen3-8B-Base
    discrete_audio:
        codec_choice: Xcodec
        codec_hf_model_tag: hf-audio/xcodec-hubert-general
        ssl_choice: null
        ssl_hf_model_tag: null
        delay_interleave: true
        stream_weights: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    continuous_audio:
        encoder_choice: huggingface
        encoder_hf_model_tag: Qwen/Qwen3-Omni-30B-A3B-Instruct
        attn_implementation: flash_attention_3
        dtype: bfloat16

model:
    model_choice: parallel
    model_hf_tag: Qwen/Qwen3-8B-Base
    model_conf:
        attn_implementation: flash_attention_3
        dtype: bfloat16
        compile_transformer_body: false
        freeze_text_embeddings: false
    activation_checkpointing: true

preprocessor:
    audio_input: continuous_audio
    audio_output: discrete_audio
    loss_region: assistant
    audio_cfg: 0.05

data_loading:
    batchfy_method: pack
    batch_size: 15000
    save_loader_state: false
    seed: 42
    num_workers: 6

trainer:
    deepspeed_config: conf/deepspeed_stage2.json
    freeze_param: [multimodal_io_dict.discrete_audio, multimodal_io_dict.continuous_audio]
    max_step: 700000
    save_interval: 1000
    log_interval: 1

```

# train.yaml 解读（这里是UALM）

关系着模型的构造

```yaml
job_type: speechlm

multimodal_io:
    text:
        tokenizer_name: Qwen/Qwen3-1.7B
    discrete_audio:
        codec_choice: Xcodec
        codec_hf_model_tag: hf-audio/xcodec-hubert-general
        ssl_choice: null
        ssl_hf_model_tag: null
        delay_interleave: true
        stream_weights: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    continuous_audio:
        encoder_choice: huggingface
        encoder_hf_model_tag: Qwen/Qwen2.5-Omni-7B
        attn_implementation: flash_attention_2
        dtype: bfloat16


# 疑问：这里audio encoder是用了qwen2.5-omni整体，还是单纯用了它的audio encoder部分（whisper-large）？

# 主体backbone是qwen3-1.7b
model:
    model_choice: parallel
    model_hf_tag: Qwen/Qwen3-1.7B
    model_conf:
        attn_implementation: flash_attention_2
        dtype: bfloat16
    activation_checkpointing: true

preprocessor:
    audio_input: continuous_audio
    audio_output: discrete_audio
    loss_region: all # 所有token都计算loss，包括audio codec token和text token。（不同于audio理解模型）
    audio_cfg: 0.05

# 输入连续音频，输出离散音频（token）。类似于codec重建。

data_loading:
    batchfy_method: pack
    batch_size: 20000
    save_loader_state: false
    seed: 42
    num_workers: 6

trainer:
    deepspeed_config: conf/deepspeed.json
    freeze_param: [multimodal_io_dict.discrete_audio, multimodal_io_dict.continuous_audio]
    max_step: 500000
    save_interval: 5000
    log_interval: 10
```