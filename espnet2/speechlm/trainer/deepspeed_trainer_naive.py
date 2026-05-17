# espnet2/speechlm/trainer/deepspeed_trainer_naive.py

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import wandb

from espnet2.speechlm.utils.data import to_device
from espnet2.speechlm.utils.model_summary import model_summary

logger = logging.getLogger(__name__)


class DeepSpeedTrainerNaive:
    def __init__(
        self,
        train_data_factory,
        valid_data_factories: Dict,
        model: nn.Module,
        resume_path: Optional[Path],
        output_dir: Path,
        trainer_args: Dict[str, Any],
    ):
        self.train_data_factory = train_data_factory
        self.valid_data_factories = valid_data_factories
        self.output_dir = Path(output_dir)
        self.trainer_args = trainer_args
        (self.output_dir / "checkpoints").mkdir(exist_ok=True, parents=True)

        self.global_step = 0
        self.max_step = trainer_args["max_step"]
        self.save_interval = trainer_args["save_interval"]
        self.log_interval = trainer_args["log_interval"]

        for t in trainer_args.get("freeze_param", []):
            for k, p in model.named_parameters():
                if k.startswith(t + ".") or k == t:
                    logger.info(f"Setting {k}.requires_grad = False")
                    p.requires_grad = False

        ds_config_path = trainer_args["deepspeed_config"]
        with open(ds_config_path, "r") as f:
            self.ds_config = json.load(f)

        logger.info(model_summary(model))

        self.model = model.cuda()
        self.dtype = self.train_dtype(self.ds_config)

        if self.dtype == torch.bfloat16:
            logger.info("Convert model parameters to bfloat16")
            self.model = self.model.to(dtype=torch.bfloat16)
        elif self.dtype == torch.float16:
            logger.info("Convert model parameters to float16")
            self.model = self.model.to(dtype=torch.float16)
        else:
            logger.info("Keep model parameters in float32")

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        optim_cfg = self.ds_config.get("optimizer", {})
        optim_params = optim_cfg.get("params", {})
        lr = optim_params.get("lr", trainer_args.get("lr", 1e-5))
        betas = tuple(optim_params.get("betas", [0.9, 0.999]))
        eps = optim_params.get("eps", 1e-8)
        weight_decay = optim_params.get("weight_decay", 0.0)

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        self.grad_accum_steps = self.ds_config.get("gradient_accumulation_steps", 1)
        self.grad_clip = self.ds_config.get("gradient_clipping", 0.0)

        wandb.config.update({"naive_deepspeed_config": self.ds_config})

        self._load_checkpoint(resume_path)

    def _load_checkpoint(self, resume_path: Optional[Path]) -> None:
        checkpoint_path = None

        if resume_path and resume_path.exists():
            checkpoint_path = resume_path
        else:
            ckpt_dir = self.output_dir / "checkpoints"
            if ckpt_dir.exists():
                checkpoints = [
                    d for d in ckpt_dir.iterdir()
                    if d.is_dir() and d.name.startswith("step_")
                ]
                if checkpoints:
                    checkpoint_path = sorted(
                        checkpoints,
                        key=lambda x: int(x.name.split("step_")[-1]),
                        reverse=True,
                    )[0]

        if checkpoint_path and checkpoint_path.is_dir():
            state_path = checkpoint_path / "naive_state.pt"
            if state_path.exists():
                state = torch.load(state_path, map_location="cpu")
                self.model.load_state_dict(state["model"])
                self.optimizer.load_state_dict(state["optimizer"])
                self.global_step = state.get("global_step", 0)
                logger.info(f"Loaded naive checkpoint: {state_path}")
            else:
                logger.info(f"No naive_state.pt in {checkpoint_path}, start fresh")

        elif checkpoint_path and checkpoint_path.is_file():
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt["module"] if "module" in ckpt else ckpt
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded model weights from: {checkpoint_path}")
        else:
            logger.info("No checkpoint found, starting from step 0")

    def _save_checkpoint(self):
        step_dir = self.output_dir / "checkpoints" / f"step_{self.global_step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            step_dir / "naive_state.pt",
        )

        # 兼容你后面 inference_ckpt 的 DeepSpeed 风格路径
        ds_like_dir = step_dir / f"global_step{self.global_step}"
        ds_like_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"module": self.model.state_dict()},
            ds_like_dir / "mp_rank_00_model_states.pt",
        )

        logger.info(f"Saved checkpoint to: {step_dir}")

    def run(self) -> None:
        while self.global_step < self.max_step:
            self.train()
            self.valid()
            self._save_checkpoint()

    def train(self) -> None:
        self.model.train()

        iterator = self.train_data_factory.build_iter(
            global_step=self.global_step,
            length=self.save_interval,
        )

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(iterator):
            iter_start = time.time()

            batch = to_device(batch, "cuda", dtype=self.dtype)
            out = self.model(**batch)

            loss = out["loss"] / self.grad_accum_steps
            loss.backward()

            do_step = (batch_idx + 1) % self.grad_accum_steps == 0

            if do_step:
                if self.grad_clip and self.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip,
                    )
                else:
                    grad_norm = torch.tensor(0.0)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                stats = out["stats"]
                stats = {f"train/{k}": float(v.detach().cpu()) for k, v in stats.items()}
                stats["train/lr"] = self.optimizer.param_groups[0]["lr"]
                stats["train/grad_norm"] = float(grad_norm)
                stats["time/iter"] = time.time() - iter_start

                wandb.log(stats, step=self.global_step)

                if self.global_step % self.log_interval == 0:
                    logger.info(f"step {self.global_step}, stats: {stats}")

                self.global_step += 1

                if self.global_step >= self.max_step:
                    break

    def valid(self) -> None:
        self.model.eval()

        for name, factory in self.valid_data_factories.items():
            iterator = factory.build_iter()
            all_stats = {}

            with torch.no_grad():
                for batch in iterator:
                    batch = to_device(batch, "cuda", dtype=self.dtype)
                    out = self.model(**batch)

                    stats = {k: float(v.detach().cpu()) for k, v in out["stats"].items()}
                    for key, value in stats.items():
                        all_stats.setdefault(key, []).append(value)

            all_stats = {
                f"val/{name}/{key}": sum(value) / len(value)
                for key, value in all_stats.items()
            }
            wandb.log(all_stats, step=self.global_step)

    def train_dtype(self, ds_config):
        if ds_config.get("bf16", {}).get("enabled", False):
            dtype = torch.bfloat16
        elif ds_config.get("fp16", {}).get("enabled", False):
            dtype = torch.float16
        else:
            dtype = torch.float32

        logger.info(f"Convert all float input data to dtype={dtype}")
        return dtype