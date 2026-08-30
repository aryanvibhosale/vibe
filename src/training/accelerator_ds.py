from __future__ import annotations

import contextlib
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data


class DeepSpeedAccelerator:

    def __init__(self, ds_config: str, seed: int = 42):
        import deepspeed  # noqa: F401 — ensures it's importable early

        self.ds_config_path = ds_config
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))

        if self.world_size > 1 and not dist.is_initialized():
            deepspeed.init_distributed(dist_backend="nccl")

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))

        self._set_seed(seed)

        # These are set after prepare_model is called
        self._engine = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def prepare_model(
        self,
        model: torch.nn.Module,
        *,
        optimizer: torch.optim.Optimizer,
        lr: float,
        weight_decay: float,
        warmup_steps: int,
        total_steps: int,
        batch_size_per_gpu: int,
        grad_accum_steps: int = 1,
        # ignored — kept for API compat with Accelerator.prepare_model
        find_unused_parameters: bool = False,
    ):
        import deepspeed
        import json

        with open(self.ds_config_path) as f:
            ds_cfg = json.load(f)

        # Fill in "auto" fields with concrete values
        ds_cfg["train_micro_batch_size_per_gpu"] = batch_size_per_gpu
        ds_cfg["gradient_accumulation_steps"] = grad_accum_steps
        ds_cfg["gradient_clipping"] = 1e9  # matches the clip_grad_norm call

        if "optimizer" in ds_cfg:
            ds_cfg["optimizer"]["params"]["lr"] = lr
            ds_cfg["optimizer"]["params"]["weight_decay"] = weight_decay

        if "scheduler" in ds_cfg:
            sch_params = ds_cfg["scheduler"]["params"]
            sch_params["warmup_max_lr"] = lr
            sch_params["warmup_num_steps"] = warmup_steps
            sch_params["total_num_steps"] = total_steps

        engine, ds_optimizer, _, ds_scheduler = deepspeed.initialize(
            model=model,
            model_parameters=[p for p in model.parameters() if p.requires_grad],
            config=ds_cfg,
        )
        self._engine = engine

        return engine, ds_optimizer, ds_scheduler

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Gradient / backward helpers
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def autocast(self, dtype=torch.bfloat16):
        yield

    def backward(self, loss: torch.Tensor):
        self._engine.backward(loss)

    def step(self, optimizer=None):
        self._engine.step()

    def update(self):
        pass

    @contextlib.contextmanager
    def no_sync(self):
        yield

    # ------------------------------------------------------------------
    # Collective helpers
    # ------------------------------------------------------------------

    def barrier(self):
        if dist.is_initialized():
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.AVG):
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------

    @staticmethod
    def unwrap(model) -> torch.nn.Module:
        if hasattr(model, "module"):
            return model.module
        return model
