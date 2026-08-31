import math
import torch

from reward_models.v2m_reward.aggregate import aggregate_scores, batch_zscore_to_r


def test_aggregate_uniform_weights():
    scores = torch.tensor([[3.0, 3.0, 3.0], [5.0, 1.0, 3.0]])
    weights = {"musicality": 1.0, "text_music_alignment": 1.0, "video_music_alignment": 1.0}
    out = aggregate_scores(scores, weights)
    assert torch.allclose(out, torch.tensor([3.0, 3.0]))


def test_aggregate_skewed_weights():
    scores = torch.tensor([[5.0, 1.0, 1.0]])
    weights = {"musicality": 1.0, "text_music_alignment": 0.0, "video_music_alignment": 0.0}
    out = aggregate_scores(scores, weights)
    assert torch.allclose(out, torch.tensor([5.0]))


def test_zscore_to_r_centers_at_half():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    r = batch_zscore_to_r(rewards, adv_clip_max=5.0)
    # mean across the batch should be ~0.5 after the affine
    assert math.isclose(r.mean().item(), 0.5, abs_tol=1e-5)
    assert (r >= 0.0).all() and (r <= 1.0).all()


def test_zscore_clamps_outliers():
    # 100 zeros + one outlier: std stays small, z-score for the outlier blows past +5
    rewards = torch.cat([torch.zeros(100), torch.tensor([1000.0])]).unsqueeze(0)
    r = batch_zscore_to_r(rewards, adv_clip_max=5.0)
    assert r[0, -1].item() == 1.0
    assert r.min().item() >= 0.0
    assert r.max().item() <= 1.0


def test_zscore_global_stats_matches_local_when_data_identical():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    flat = rewards.reshape(-1)
    r_local = batch_zscore_to_r(rewards, adv_clip_max=5.0)
    r_global = batch_zscore_to_r(
        rewards, adv_clip_max=5.0,
        global_mean=flat.mean(), global_std=flat.std(),
    )
    assert torch.allclose(r_local, r_global, atol=1e-6)


def test_zscore_global_stats_shifts_when_global_differs():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    r_local = batch_zscore_to_r(rewards, adv_clip_max=5.0)
    r_with_high_global_mean = batch_zscore_to_r(
        rewards, adv_clip_max=5.0,
        global_mean=torch.tensor(10.0), global_std=torch.tensor(1.0),
    )
    assert r_with_high_global_mean.mean() < r_local.mean()
