from typing import Any, Dict, List, Optional, Sequence

import torch

from .inference import CMIModelInference


class SoftRewardTensor:

    #: axis order returned by score_prompt_group
    AXES = ("alignment", "quality")

    def __init__(
        self,
        checkpoint: str,
        config: Optional[str] = None,
        device: str = "cuda:0",
    ):
        self.model = CMIModelInference(checkpoint=checkpoint, config=config, device=device)
        self.device = device
        self._stat_calls = 0
        self._stat_prompts = 0
        self._stat_failures = 0
        self._last_raw_scores: List[Dict[str, float]] = []

    # ----------------- one-shot -----------------

    def score_tensor(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        text: str = "",
        **_: Any,
    ) -> Dict[str, float]:
        result = self.model.score_one(
            waveform=self._ensure_mono(waveform),
            sample_rate=sample_rate,
            text=text,
        )
        self._last_raw_scores = [result]
        self._stat_calls += 1
        return result

    # ----------------- group (one prompt, G candidates) -----------------

    def score_prompt_group(
        self,
        waveforms: List[torch.Tensor],
        text: Any = "",
        sample_rate: int = 48000,
        video_path: Optional[str] = None,   # unused; keeps the call shape uniform
        **_: Any,
    ) -> torch.Tensor:
        n = len(waveforms)
        if isinstance(text, (list, tuple)):
            if len(text) != n:
                raise ValueError(f"got {len(text)} captions for {n} candidates")
            captions: Sequence[str] = text
        else:
            captions = [text] * n

        results: List[Dict[str, float]] = []
        out = torch.empty(n, 2, dtype=torch.float32)
        for i, (wav, cap) in enumerate(zip(waveforms, captions)):
            r = self.model.score_one(
                waveform=self._ensure_mono(wav), sample_rate=sample_rate, text=cap
            )
            results.append(r)
            self._stat_calls += 1
            out[i, 0] = r["alignment"]
            out[i, 1] = r["quality"]
        self._last_raw_scores = results
        self._stat_prompts += 1
        return out

    def score_prompt_group_scalar(
        self,
        waveforms: List[torch.Tensor],
        text: Any = "",
        sample_rate: int = 48000,
        weights: Optional[Dict[str, float]] = None,
        **_: Any,
    ) -> torch.Tensor:
        from .aggregate import aggregate_scores
        per_axis = self.score_prompt_group(waveforms, text, sample_rate)
        return aggregate_scores(per_axis, weights=weights)

    # ----------------- diagnostics -----------------

    def stats(self) -> Dict[str, int]:
        return {
            "calls": self._stat_calls,
            "prompts": self._stat_prompts,
            "parse_failures": self._stat_failures,
        }

    def reset_stats(self):
        self._stat_calls = self._stat_prompts = self._stat_failures = 0

    def last_raw_scores(self) -> List[Dict[str, float]]:
        return list(self._last_raw_scores)

    # ----------------- helpers -----------------

    @staticmethod
    def _ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
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
        return f"SoftRewardTensor(checkpoint={self.model.checkpoint!r}, device={self.device})"
