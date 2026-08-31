from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from ..base import RewardConfigurator
from . import aggregate as _aggregate
from . import prompts as _prompts
from . import schema as _schema
from .wrapper import HardRewardTensor


class HardVerifiableConfigurator(RewardConfigurator):

    model_type = "hard_verifiable"
    requires_model_load = False          # nothing to download or quantize
    # The trainers add w_tempo*t + w_key*k onto the primary reward, so the
    # aggregator fires by default and returns that contribution.
    default_requires_aggregation = True
    axes = _schema.AXES                  # ("tempo", "key")

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        *,
        reward_config: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
        requires_aggregation: Optional[bool] = None,
        sample_rate: int = 48000,
    ):
        super().__init__(
            model_path=model_path,
            device=device,
            weights=weights,
            requires_aggregation=requires_aggregation,
            sample_rate=sample_rate,
        )
        #: tempo_sigma / tempo_ramp_width / key_sigma
        self.reward_config = dict(reward_config or {})

    # --------------------------------------------------------------- config

    @classmethod
    def _from_config_dict(cls, rm_cfg: Dict[str, Any], device: str) -> "HardVerifiableConfigurator":
        src = rm_cfg.get("reward_model", rm_cfg) if "reward_model" in rm_cfg else rm_cfg
        return cls(
            device="cpu",
            reward_config=rm_cfg.get("hard_verifiable_reward_config", {}),
            weights={
                "tempo": float(src.get("reward_weight_tempo", 0.0)),
                "key":   float(src.get("reward_weight_key",   0.0)),
            },
            requires_aggregation=rm_cfg.get("aggregate"),
        )

    @property
    def enabled(self) -> bool:
        w = self.weights or {}
        return (w.get("tempo", 0.0) > 0.0) or (w.get("key", 0.0) > 0.0)

    # -------------------------------------------------------------- loading

    def load(self) -> None:
        if self._loaded:
            return
        # HardRewardTensor -> HardVerifiableRewards. Nothing to download.
        self.model = HardRewardTensor(self.reward_config)
        self._loaded = True

    # -------------------------------------------------------------- scoring

    def score_one(
        self,
        waveform: torch.Tensor,
        attributes: Optional[Dict[str, Any]],
        sample_rate: Optional[int] = None,
        **_: Any,
    ) -> Dict[str, float]:
        self._ensure_loaded()
        out = self.model.score_tensor(
            waveform=waveform,
            sample_rate=sample_rate or self.sample_rate,
            attributes=attributes,
        )
        self._sync_stats()
        return out

    def score_group(
        self,
        waveforms: List[torch.Tensor],
        attributes: Any = None,
        sample_rate: Optional[int] = None,
        **_: Any,
    ) -> torch.Tensor:
        self._ensure_loaded()
        n = len(waveforms)
        if isinstance(attributes, (list, tuple)):
            if len(attributes) != n:
                raise ValueError(f"got {len(attributes)} attribute rows for {n} candidates")
            attrs: Sequence[Optional[Dict[str, Any]]] = attributes
        else:
            attrs = [attributes] * n

        out = self.model.score_prompt_group(
            waveforms=waveforms,
            attributes=list(attrs),
            sample_rate=sample_rate or self.sample_rate,
        )
        self._sync_stats()
        return out

    def aggregate(self, per_axis: torch.Tensor) -> torch.Tensor:
        return _aggregate.aggregate_scores(per_axis, weights=self.weights)

    # ------------------------------------------------------ schema / prompts

    @property
    def schema(self):
        return _schema

    @property
    def prompts(self):
        return _prompts

    # ---------------------------------------------------------- diagnostics

    def _sync_stats(self) -> None:
        if self.model is None:
            return
        st = self.model.stats()
        self._stat_calls, self._stat_prompts = st["calls"], st["prompts"]
        self._stat_failures = st["parse_failures"]

    def stats(self) -> Dict[str, int]:
        self._sync_stats()
        return super().stats()

    def reset_stats(self) -> None:
        super().reset_stats()
        if self.model is not None:
            self.model.reset_stats()

    def last_raw_scores(self):
        return self.model.last_raw_scores() if self.model else []


#: uniform alias
HardRewardConfigurator = HardVerifiableConfigurator
