from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

log = logging.getLogger(__name__)


class RewardConfigurator(ABC):

    #: identifier matching `reward_model.type` in config YAML
    model_type: str = "abstract"
    #: False for rule-based rewards that have no weights to load
    requires_model_load: bool = True
    #: default for whether compute_reward() collapses per-axis output to a scalar
    default_requires_aggregation: bool = True
    #: axis names produced by `score_group`, in order
    axes: Sequence[str] = ()

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda:0",
        *,
        weights: Optional[Dict[str, float]] = None,
        requires_aggregation: Optional[bool] = None,
        sample_rate: int = 48000,
        **kwargs: Any,
    ):
        self.model_path = model_path
        self.device = device
        self.weights = dict(weights) if weights else None
        self.requires_aggregation = (
            self.default_requires_aggregation
            if requires_aggregation is None
            else bool(requires_aggregation)
        )
        self.sample_rate = sample_rate
        self.options: Dict[str, Any] = dict(kwargs)

        self.model: Any = None
        self._loaded = False
        self._stat_calls = 0
        self._stat_prompts = 0
        self._stat_failures = 0

    # ------------------------------------------------------------------ config

    @classmethod
    def from_config(
        cls,
        rm_cfg: Dict[str, Any],
        device: str,
        *,
        rank: int = 0,
        barrier: Optional[Callable[[], None]] = None,
        autoload: bool = True,
    ) -> "RewardConfigurator":
        rm_type = rm_cfg.get("type", cls.model_type)
        target = cls if cls is not RewardConfigurator else _resolve(rm_type)
        inst = target._from_config_dict(rm_cfg, device)
        if autoload and inst.requires_model_load:
            inst.load_rank0_first(rank=rank, barrier=barrier)
        elif autoload:
            inst.load()
        return inst

    @classmethod
    @abstractmethod
    def _from_config_dict(cls, rm_cfg: Dict[str, Any], device: str) -> "RewardConfigurator":
        pass

    # ------------------------------------------------------------------ loading

    @abstractmethod
    def load(self) -> None:
        pass

    def load_rank0_first(self, rank: int = 0, barrier: Optional[Callable[[], None]] = None) -> None:
        if rank == 0:
            self.load()
        if barrier is not None:
            barrier()
        if rank != 0:
            self.load()

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------ scoring

    @abstractmethod
    def score_group(self, **kwargs: Any) -> torch.Tensor:
        pass

    def compute_reward(self, **kwargs: Any) -> torch.Tensor:
        self._ensure_loaded()
        per_axis = self.score_group(**kwargs)
        if not self.requires_aggregation:
            return per_axis
        return self.aggregate(per_axis)

    @abstractmethod
    def aggregate(self, per_axis: torch.Tensor) -> torch.Tensor:
        pass

    def components(self, per_axis: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: per_axis[..., i] for i, name in enumerate(self.axes)}

    # ------------------------------------------------------------ normalization

    def normalize(
        self,
        rewards: torch.Tensor,
        mode: str = "batch_zscore",
        adv_clip_max: float = 5.0,
        global_mean: Optional[torch.Tensor] = None,
        global_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Imported lazily: the judge package __init__ pulls in transformers, and
        # base.py must stay importable without it (and without a circular import).
        from .v2m_reward.aggregate import batch_zscore_to_r

        if mode == "batch_zscore":
            return batch_zscore_to_r(rewards, adv_clip_max=adv_clip_max)
        if mode == "global_zscore":
            if global_mean is None or global_std is None:
                raise ValueError("global_zscore requires global_mean and global_std")
            return batch_zscore_to_r(
                rewards, adv_clip_max=adv_clip_max,
                global_mean=global_mean, global_std=global_std,
            )
        raise ValueError(
            f"unknown normalization {mode!r}; per_prompt_running is owned by the "
            f"trainer's stat tracker"
        )

    # ------------------------------------------------- schema / prompts modules

    @property
    def schema(self):
        raise NotImplementedError

    @property
    def prompts(self):
        return None

    # ------------------------------------------------------------- diagnostics

    def stats(self) -> Dict[str, int]:
        return {
            "calls": self._stat_calls,
            "prompts": self._stat_prompts,
            "failures": self._stat_failures,
        }

    def reset_stats(self) -> None:
        self._stat_calls = self._stat_prompts = self._stat_failures = 0

    def last_raw_scores(self) -> List[Any]:
        return []

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
        w = waveform.detach()
        if w.dim() == 3:
            w = w.squeeze(0)
        if w.dim() == 2:
            if w.shape[0] <= 4 and w.shape[1] > w.shape[0]:
                w = w.mean(dim=0) if w.shape[0] > 1 else w.squeeze(0)
            else:
                w = w.mean(dim=1) if w.shape[1] > 1 else w.squeeze(1)
        if w.dim() != 1:
            raise ValueError(f"expected 1D waveform after mono-flatten, got {w.shape}")
        return w.to(torch.float32).cpu()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(model_type={self.model_type!r}, "
            f"model_path={self.model_path!r}, device={self.device!r}, "
            f"loaded={self._loaded}, aggregate={self.requires_aggregation})"
        )


# ---------------------------------------------------------------- registry

_REGISTRY: Dict[str, str] = {
    # reward_model.type    ->  "module:ClassName"
    "multimodal_llm_judge": "reward_models.v2m_reward.configurator:JudgeConfigurator",
    "cmi_rm_v2m":           "reward_models.soft_reward.configurator:CMIConfigurator",
    "cmi_rm":               "reward_models.soft_reward.configurator:CMIConfigurator",
    "hard_verifiable":      "reward_models.hard_reward.configurator:HardVerifiableConfigurator",
}


def _resolve(rm_type: str):
    from importlib import import_module
    if rm_type not in _REGISTRY:
        raise ValueError(
            f"unknown reward_model.type {rm_type!r}; known types: {sorted(_REGISTRY)}"
        )
    mod_name, cls_name = _REGISTRY[rm_type].split(":")
    return getattr(import_module(mod_name), cls_name)


def build_reward_configurator(
    rm_cfg: Dict[str, Any],
    device: str,
    *,
    rank: int = 0,
    barrier: Optional[Callable[[], None]] = None,
    autoload: bool = True,
) -> RewardConfigurator:
    return RewardConfigurator.from_config(
        rm_cfg, device, rank=rank, barrier=barrier, autoload=autoload
    )


def available_reward_types() -> List[str]:
    return sorted(_REGISTRY)
