from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from ..base import RewardConfigurator
from . import aggregate as _aggregate
from . import prompts as _prompts
from . import schema as _schema
from .wrapper import SoftRewardTensor


class CMIConfigurator(RewardConfigurator):

    model_type = "cmi_rm_v2m"
    requires_model_load = True
    # The trainers compose alignment/quality with configured weights before
    # normalization, so the aggregator fires by default.
    default_requires_aggregation = True
    axes = _schema.AXES  # ("alignment", "quality")

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        *,
        cmi_config: Optional[str] = None,
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
        #: path to the CMI-RM config.yaml that pairs with the checkpoint
        self.cmi_config = cmi_config

    # --------------------------------------------------------------- config

    @classmethod
    def _from_config_dict(cls, rm_cfg: Dict[str, Any], device: str) -> "CMIConfigurator":
        # Accept both the V2M block (`reward_model.cmi_checkpoint`) and the TTM
        # top-level spelling (`cmi_checkpoint`), which are the two live layouts.
        ckpt = rm_cfg.get("cmi_checkpoint") or rm_cfg.get("checkpoint")
        if ckpt is None:
            raise KeyError("CMI-RM requires `cmi_checkpoint` in the reward_model block")
        return cls(
            model_path=ckpt,
            device=device,
            cmi_config=rm_cfg.get("cmi_config") or rm_cfg.get("config"),
            weights={
                "alignment": float(rm_cfg.get("reward_weight_alignment", 0.5)),
                "quality":   float(rm_cfg.get("reward_weight_quality",   0.5)),
            },
            requires_aggregation=rm_cfg.get("aggregate"),
        )

    # -------------------------------------------------------------- loading

    def load(self) -> None:
        if self._loaded:
            return
        # SoftRewardTensor -> CMIModelInference -> external cmi_rm, imported
        # lazily so a run that does not select this reward never needs it.
        self.model = SoftRewardTensor(
            checkpoint=self.model_path,
            config=self.cmi_config,
            device=self.device,
        )
        self._loaded = True

    # -------------------------------------------------------------- scoring

    def score_one(
        self,
        waveform: torch.Tensor,
        text: str,
        sample_rate: Optional[int] = None,
        **_: Any,
    ) -> Dict[str, float]:
        self._ensure_loaded()
        scores = self.model.score_tensor(
            waveform=waveform,
            sample_rate=sample_rate or self.sample_rate,
            text=_prompts.build_conditioning_text(text),
        )
        self._sync_stats()
        return {"alignment": float(scores["alignment"]), "quality": float(scores["quality"])}

    def score_group(
        self,
        waveforms: List[torch.Tensor],
        text: Any = "",
        sample_rate: Optional[int] = None,
        **_: Any,
    ) -> torch.Tensor:
        self._ensure_loaded()
        n = len(waveforms)
        if isinstance(text, (list, tuple)):
            if len(text) != n:
                raise ValueError(f"got {len(text)} captions for {n} candidates")
            captions: Sequence[str] = text
        else:
            captions = [text] * n

        out = self.model.score_prompt_group(
            waveforms=waveforms,
            text=[_prompts.build_conditioning_text(c) for c in captions],
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
SoftRewardConfigurator = CMIConfigurator
