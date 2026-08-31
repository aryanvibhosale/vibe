from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

log = logging.getLogger(__name__)


class CMIModelInference:

    def __init__(
        self,
        checkpoint: str,
        config: Optional[str] = None,
        device: str = "cuda:0",
    ):
        self.model_id = checkpoint
        self.checkpoint = checkpoint
        self.config = config
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        try:
            from cmi_rm import CMIRewardTensor as _ExternalCMI
        except ImportError as e:                                    # pragma: no cover
            raise ImportError(
                "cmi_rm is not installed. Its weights are CC-BY-NC-4.0 and are not "
                "redistributed with VIBE; obtain CMI-RewardBench separately and put "
                "cmi_rm on PYTHONPATH. See docs/INSTALL.md."
            ) from e
        log.info("loading CMI-RM %s on %s", self.checkpoint, self.device)
        return _ExternalCMI(
            checkpoint=self.checkpoint,
            config=self.config,
            device=self.device,
        )

    def score_one(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        text: str = "",
        **_: Any,
    ) -> Dict[str, float]:
        scores = self.model.score_tensor(
            waveform=waveform,
            sample_rate=sample_rate,
            text=text,
        )
        return {
            "alignment": float(scores["alignment"]),
            "quality": float(scores["quality"]),
        }

    def __repr__(self) -> str:
        return f"CMIModelInference(checkpoint={self.checkpoint!r}, device={self.device!r})"
