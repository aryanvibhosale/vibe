from typing import Dict, Optional

import torch

_DEFAULT_WEIGHTS = {"alignment": 0.5, "quality": 0.5}


def aggregate_scores(
    scores: torch.Tensor,
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    w = weights or _DEFAULT_WEIGHTS
    v = torch.tensor(
        [w.get("alignment", 0.5), w.get("quality", 0.5)],
        dtype=scores.dtype,
        device=scores.device,
    )
    return (scores * v).sum(dim=-1)
