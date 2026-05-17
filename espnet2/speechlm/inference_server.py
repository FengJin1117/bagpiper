import os

from fastapi import FastAPI
from pydantic import BaseModel

import uvicorn

from espnet2.speechlm.service.model_worker import init_model
from espnet2.speechlm.service.infer_core import infer_audio


TRAIN_CONFIG = os.environ.get(
    "TRAIN_CONFIG",
    "conf/train.yaml",
)

INFERENCE_CONFIG = os.environ.get(
    "INFERENCE_CONFIG",
    "conf/inference.yaml",
)

CHECKPOINT = os.environ.get(
    "CHECKPOINT",
    "exp/opuslm_v2_stage2_pretrain_base/checkpoints/step_260000/global_step259988/mp_rank_00_model_states.pt",
)


app = FastAPI()


class InferRequest(BaseModel):
    audio_path: str


@app.on_event("startup")
def startup_event():

    init_model(
        train_config_path=TRAIN_CONFIG,
        inference_config_path=INFERENCE_CONFIG,
        checkpoint_path=CHECKPOINT,
    )


@app.get("/health")
def health():

    return {
        "status": "ok",
    }

# 推理接口
@app.post("/infer")
def infer(req: InferRequest):

    text = infer_audio(
        req.audio_path
    )

    return {
        "text": text,
    }


if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        # reload=True, # 方便开发调试
    )