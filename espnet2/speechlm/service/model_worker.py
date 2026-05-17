import random
from pathlib import Path

import numpy as np
import torch
import yaml

from espnet2.speechlm.model import _all_job_types


MODEL = None
PREPROCESSOR = None
INFERENCE_CONFIG = None
DTYPE = None


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def init_model(
    train_config_path,
    inference_config_path,
    checkpoint_path,
    dtype="bfloat16",
    seed=42,
):

    global MODEL
    global PREPROCESSOR
    global INFERENCE_CONFIG
    global DTYPE

    if MODEL is not None:
        return

    assert torch.cuda.is_available()

    set_seed(seed)

    print("[INFO] Loading configs...")

    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)

    with open(inference_config_path, "r") as f:
        inference_config = yaml.safe_load(f)

    print("[INFO] Building job template...")

    job_template_class = _all_job_types[
        train_config["job_type"]
    ]

    job_template = job_template_class(
        train_config,
        is_train=False,
    )

    print("[INFO] Building preprocessor...")

    preprocessor = job_template.build_preprocessor()

    print("[INFO] Building model...")

    model = job_template.build_model()

    print("[INFO] Loading checkpoint...")

    model = load_checkpoint(
        model,
        checkpoint_path,
    )

    print("[INFO] Preparing inference...")

    model.prepare_inference()

    torch_dtype = getattr(torch, dtype)

    model = model.to(
        device="cuda",
        dtype=torch_dtype,
    ).eval()

    MODEL = model
    PREPROCESSOR = preprocessor
    INFERENCE_CONFIG = inference_config
    DTYPE = torch_dtype

    print("[INFO] Model initialized successfully")


def get_model():
    return MODEL


def get_preprocessor():
    return PREPROCESSOR


def get_inference_config():
    return INFERENCE_CONFIG


def get_dtype():
    return DTYPE