from typing import Any, Dict, List, Optional, Sequence

import torch

from .inference import HardVerifiableRewards
from .prompts import build_targets


class HardRewardTensor:

    #: axis order returned by score_prompt_group
    AXES = ("tempo", "key")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.model = HardVerifiableRewards(dict(config or {}))
        self.device = "cpu"
        self._stat_calls = 0
        self._stat_prompts = 0
        self._stat_failures = 0
        self._last_raw_scores: List[Dict[str, float]] = []

    # ----------------- one-shot -----------------

    def score_tensor(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        attributes: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, float]:
        targets = build_targets(attributes or {})
        if targets is None:
            self._stat_failures += 1
            result = {"tempo": float("nan"), "key": float("nan")}
            self._last_raw_scores = [result]
            return result

        wav = self._ensure_mono(waveform).numpy() if torch.is_tensor(waveform) else waveform
        out = self.model.compute_rewards(waveform=wav, sample_rate=sample_rate, **targets)
        self._stat_calls += 1
        t, k = out.get("tempo"), out.get("key")
        result = {
            "tempo": float(t) if t is not None else float("nan"),
            "key": float(k) if k is not None else float("nan"),
        }
        self._last_raw_scores = [result]
        return result

    # ----------------- group (one prompt, G candidates) -----------------

    def score_prompt_group(
        self,
        waveforms: List[torch.Tensor],
        attributes: Any = None,
        sample_rate: int = 48000,
        **_: Any,
    ) -> torch.Tensor:
        n = len(waveforms)
        if isinstance(attributes, (list, tuple)):
            if len(attributes) != n:
                raise ValueError(f"got {len(attributes)} attribute rows for {n} candidates")
            attrs: Sequence[Optional[Dict[str, Any]]] = attributes
        else:
            attrs = [attributes] * n

        results: List[Dict[str, float]] = []
        out = torch.empty(n, 2, dtype=torch.float32)
        for i, (wav, a) in enumerate(zip(waveforms, attrs)):
            r = self.score_tensor(waveform=wav, sample_rate=sample_rate, attributes=a)
            results.append(r)
            out[i, 0] = r["tempo"]
            out[i, 1] = r["key"]
        self._last_raw_scores = results
        self._stat_prompts += 1
        return out

    def score_prompt_group_scalar(
        self,
        waveforms: List[torch.Tensor],
        attributes: Any = None,
        sample_rate: int = 48000,
        weights: Optional[Dict[str, float]] = None,
        **_: Any,
    ) -> torch.Tensor:
        from .aggregate import aggregate_scores
        per_axis = self.score_prompt_group(waveforms, attributes, sample_rate)
        return aggregate_scores(per_axis, weights=weights)

    # ----------------- diagnostics -----------------

    def stats(self) -> Dict[str, int]:
        return {
            "calls": self._stat_calls,
            "prompts": self._stat_prompts,
            # rows skipped for missing tempo/key/scale attributes
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
        return f"HardRewardTensor(config={self.model.config!r})"
