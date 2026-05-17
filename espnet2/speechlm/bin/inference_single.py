#!/usr/bin/env python3

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from espnet2.speechlm.model import _all_job_types
from espnet2.speechlm.utils.data import to_device


def load_checkpoint(model, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    state_dict = checkpoint["module"]

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    return model


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--train-config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--inference-config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
    )

    # 生成模型中常用
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser


@torch.no_grad()
def main():

    args = get_parser().parse_args()

    assert torch.cuda.is_available()

    set_seed(args.seed)

    # =========================================================
    # load audio
    # =========================================================

    print(f"Loading audio: {args.audio}")

    audio, sr = sf.read(args.audio)

    # convert to float32
    audio = audio.astype(np.float32)

    # ensure shape = [C, T]
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    elif audio.ndim == 2:
        # soundfile returns [T, C]
        audio = audio.transpose(1, 0)

    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")

    print(f"Audio shape: {audio.shape}")
    print(f"Sample rate: {sr}")

    # =========================================================
    # build sample
    # =========================================================

    key = (
        "audio_to_text",
        "custom",
        "sample_001",
    )

    data_dict = {
        "audio1": (
            audio,
            sr,
        )
    }

    # =========================================================
    # load configs
    # =========================================================

    with open(args.train_config, "r") as f:
        train_config = yaml.safe_load(f)

    with open(args.inference_config, "r") as f:
        inference_config = yaml.safe_load(f)

    # =========================================================
    # build template
    # =========================================================

    job_template_class = _all_job_types[
        train_config["job_type"]
    ]

    job_template = job_template_class(
        train_config,
        is_train=False,
    )

    # =========================================================
    # build preprocessor
    # =========================================================

    print("Building preprocessor...")

    preprocessor = job_template.build_preprocessor()

    # =========================================================
    # collate
    # =========================================================

    print("Running preprocessing...")

    batch = preprocessor.collate_fn([
        (key, data_dict)
    ])

    # =========================================================
    # build model
    # =========================================================

    print("Building model...")

    model = job_template.build_model()

    print("Loading checkpoint...")

    model = load_checkpoint(
        model,
        args.model_checkpoint,
    )

    model.prepare_inference()

    dtype = getattr(torch, args.dtype)

    model = model.to(
        device="cuda",
        dtype=dtype,
    ).eval()


    # =========================================================
    # move to device
    # =========================================================

    batch = to_device(
        batch,
        "cuda",
        dtype=dtype,
    )

    # remove metadata
    batch.pop("keys", None)

    # =========================================================
    # inference
    # =========================================================

    print("Running inference...")

    messages, _ = model.inference(
        inference_config,
        **batch,
    )

    # =========================================================
    # print outputs
    # =========================================================

    print("\n========== RESULT ==========\n")

    for role, modality, content in messages:

        print(f"[{role}] [{modality}]")

        if modality == "text":

            print(content)

        elif modality == "audio":

            print("<generated audio>")

        else:

            print(content)

        print()


if __name__ == "__main__":
    main()