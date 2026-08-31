from typing import Dict, Optional

import torch

_DEFAULT_WEIGHTS = {"tempo": 0.0, "key": 0.0}


def aggregate_scores(
    scores: torch.Tensor,
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    w = weights or _DEFAULT_WEIGHTS
    v = torch.tensor(
        [w.get("tempo", 0.0), w.get("key", 0.0)],
        dtype=scores.dtype,
        device=scores.device,
    )
    # NaN -> 0 contribution (skipped axis), not NaN propagation.
    return (torch.nan_to_num(scores, nan=0.0) * v).sum(dim=-1)
