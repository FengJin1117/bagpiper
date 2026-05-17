import numpy as np
import soundfile as sf
import torch

from espnet2.speechlm.utils.data import to_device

from espnet2.speechlm.service.model_worker import (
    get_model,
    get_preprocessor,
    get_inference_config,
    get_dtype,
)


@torch.no_grad()
def infer_audio(audio_path):

    model = get_model()
    preprocessor = get_preprocessor()
    inference_config = get_inference_config()
    dtype = get_dtype()

    # =========================================================
    # load audio
    # =========================================================

    audio, sr = sf.read(audio_path)

    audio = audio.astype(np.float32)

    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    elif audio.ndim == 2:
        audio = audio.transpose(1, 0)

    else:
        raise ValueError(f"Unexpected audio shape: {audio.shape}")

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
    # preprocess
    # =========================================================

    batch = preprocessor.collate_fn([
        (key, data_dict)
    ])

    batch = to_device(
        batch,
        "cuda",
        dtype=dtype,
    )

    batch.pop("keys", None)

    # =========================================================
    # inference
    # =========================================================

    messages, _ = model.inference(
        inference_config,
        **batch,
    )

    outputs = []

    for role, modality, content in messages:

        if modality == "text":

            # print("CONTENT =", content)
            # print("TYPE =", type(content))

            if isinstance(content, list):

                content = " ".join(map(str, content))

            outputs.append(str(content))

    return "\n".join(outputs)
