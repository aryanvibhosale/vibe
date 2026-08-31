from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ..base import RewardConfigurator
from . import aggregate as _aggregate
from . import prompts as _prompts
from . import schema as _schema
from .inference import JudgeRawScores
from .wrapper import JudgeRewardTensor


class JudgeConfigurator(RewardConfigurator):

    model_type = "multimodal_llm_judge"
    requires_model_load = True
    # The trainer collapses the judge's three Likert axes with aggregate_scores
    # before normalization, so aggregation is on by default here.
    default_requires_aggregation = True
    axes = ("musicality", "text_music_alignment", "video_music_alignment")

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        *,
        dtype: str = "bfloat16",
        load_in_4bit: bool = True,
        score_mode: str = "expected_value",
        n_video_frames: int = 4,
        max_new_tokens: int = 96,
        nan_on_failure: bool = True,
        max_memory: Optional[dict] = None,
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
        # Efficiency knobs, forwarded verbatim to JudgeModelInference.
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit
        self.score_mode = score_mode
        self.n_video_frames = n_video_frames
        self.max_new_tokens = max_new_tokens
        self.nan_on_failure = nan_on_failure
        self.max_memory = max_memory

    # --------------------------------------------------------------- config

    @classmethod
    def _from_config_dict(cls, rm_cfg: Dict[str, Any], device: str) -> "JudgeConfigurator":
        return cls(
            model_path=rm_cfg["model_id"],
            device=device,
            dtype=rm_cfg.get("dtype", "bfloat16"),
            load_in_4bit=rm_cfg.get("load_in_4bit", True),
            score_mode=rm_cfg.get("score_mode", "expected_value"),
            n_video_frames=rm_cfg.get("n_video_frames", 4),
            max_new_tokens=rm_cfg.get("max_new_tokens", 96),
            nan_on_failure=rm_cfg.get("nan_on_failure", True),
            max_memory=rm_cfg.get("max_memory"),
            weights=rm_cfg.get("axis_weights"),
            requires_aggregation=rm_cfg.get("aggregate"),
        )

    # -------------------------------------------------------------- loading

    def load(self) -> None:
        if self._loaded:
            return
        self.model = JudgeRewardTensor(
            model_id=self.model_path,
            device=self.device,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit,
            score_mode=self.score_mode,
            n_video_frames=self.n_video_frames,
            max_new_tokens=self.max_new_tokens,
            nan_on_failure=self.nan_on_failure,
            max_memory=self.max_memory,
        )
        self._loaded = True

    # -------------------------------------------------------------- scoring

    def score_group(
        self,
        video_path: str,
        text: str,
        waveforms: List[torch.Tensor],
        sample_rate: Optional[int] = None,
        **_: Any,
    ) -> torch.Tensor:
        self._ensure_loaded()
        out = self.model.score_prompt_group(
            video_path=video_path,
            text=text,
            waveforms=waveforms,
            sample_rate=sample_rate or self.sample_rate,
        )
        s = self.model.stats()
        self._stat_calls, self._stat_prompts = s["calls"], s["prompts"]
        self._stat_failures = s["parse_failures"]
        return out

    def score_one(
        self,
        waveform: torch.Tensor,
        video_path: str,
        text: str = "",
        sample_rate: Optional[int] = None,
    ) -> Dict[str, float]:
        self._ensure_loaded()
        return self.model.score_tensor(
            waveform=waveform,
            sample_rate=sample_rate or self.sample_rate,
            video_path=video_path,
            text=text,
        )

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

    def stats(self) -> Dict[str, int]:
        if self.model is None:
            return super().stats()
        s = self.model.stats()
        return {"calls": s["calls"], "prompts": s["prompts"], "failures": s["parse_failures"]}

    def reset_stats(self) -> None:
        super().reset_stats()
        if self.model is not None:
            self.model.reset_stats()

    def last_raw_outputs(self) -> List[str]:
        return self.model.last_raw_outputs() if self.model else []

    def last_raw_scores(self) -> List[JudgeRawScores]:
        return list(self.model._last_raw_scores) if self.model else []


#: uniform alias
V2MRewardConfigurator = JudgeConfigurator
