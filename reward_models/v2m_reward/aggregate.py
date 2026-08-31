from typing import Dict, Optional
import torch


_DEFAULT_WEIGHTS = {
    "musicality": 1.0,
    "text_music_alignment": 1.0,
    "video_music_alignment": 1.0,
}


def aggregate_scores(
    scores: torch.Tensor,
    weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    weights = weights or _DEFAULT_WEIGHTS
    w = torch.tensor(
        [
            weights["musicality"],
            weights["text_music_alignment"],
            weights["video_music_alignment"],
        ],
        dtype=scores.dtype,
        device=scores.device,
    )
    w = w / (w.sum() + 1e-12)
    return (scores * w).sum(dim=-1)


def batch_zscore_to_r(
    rewards: torch.Tensor,
    adv_clip_max: float = 5.0,
    global_mean: Optional[torch.Tensor] = None,
    global_std: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if global_mean is None or global_std is None:
        flat = rewards.reshape(-1)
        # nanmean/nanstd so a few failed candidates don't poison the batch.
        mask = ~torch.isnan(flat)
        if mask.any():
            valid = flat[mask]
            mean = valid.mean()
            std = valid.std()
        else:
            mean = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
            std = torch.ones((), device=rewards.device, dtype=rewards.dtype)
    else:
        mean = global_mean.to(rewards.device)
        std = global_std.to(rewards.device)
    advantages = (rewards - mean) / (std + 1e-8)
    advantages = advantages.clamp(-adv_clip_max, adv_clip_max)
    r = (advantages / adv_clip_max) / 2.0 + 0.5
    return r.clamp(0.0, 1.0)
