import torch

from reward_models.v2m_reward import aggregate_scores, batch_zscore_to_r
from reward_models.v2m_reward.inference import JudgeRawScores


class _DeterministicJudge:

    def score_prompt_group(self, video_path, text, waveforms, sample_rate):
        out = []
        for g in range(len(waveforms)):
            # encode b via text length, g via index
            b_proxy = len(text)
            score = 1.0 + ((b_proxy + g) % 5)  # in [1,5]
            out.append(torch.tensor([score, score, score]))
        return torch.stack(out)


def test_phase_a_fills_all_rewards_correctly():
    judge = _DeterministicJudge()
    weights = {"musicality": 1.0, "text_music_alignment": 1.0, "video_music_alignment": 1.0}
    local_B, G = 3, 4
    prompts = [f"p{'!' * i}" for i in range(local_B)]  # different lengths
    video_paths = [f"/v/{i}.mp4" for i in range(local_B)]
    wavs = [[torch.randn(48000) for _ in range(G)] for _ in range(local_B)]

    all_rewards = torch.zeros(local_B, G)
    for b in range(local_B):
        raw = judge.score_prompt_group(video_paths[b], prompts[b], wavs[b], 48000)
        agg = aggregate_scores(raw, weights)
        all_rewards[b] = agg

    # Each row should be the candidate-indexed Likert mod 5 (since all 3 axes match
    # and weights are uniform, the aggregate equals the raw score).
    for b in range(local_B):
        for g in range(G):
            expected = 1.0 + ((len(prompts[b]) + g) % 5)
            assert all_rewards[b, g].item() == expected


def test_phase_b_batch_zscore_in_range_and_centered():
    rewards = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                            [2.0, 3.0, 4.0, 5.0],
                            [1.0, 1.0, 5.0, 5.0]])
    r = batch_zscore_to_r(rewards, adv_clip_max=5.0)
    assert r.shape == rewards.shape
    assert (r >= 0).all() and (r <= 1).all()
    # The affine centers the global distribution at 0.5
    assert abs(r.mean().item() - 0.5) < 1e-4


def test_phase_b_global_vs_local_zscore_shape_consistency():
    rewards = torch.randn(4, 8)
    r_local = batch_zscore_to_r(rewards, adv_clip_max=5.0)
    fake_global_mean = rewards.mean()
    fake_global_std = rewards.std()
    r_global = batch_zscore_to_r(
        rewards, adv_clip_max=5.0,
        global_mean=fake_global_mean, global_std=fake_global_std,
    )
    assert r_local.shape == r_global.shape == rewards.shape
    # When global stats == local stats, results match exactly.
    assert torch.allclose(r_local, r_global, atol=1e-6)


def test_phase_b_constant_rewards_collapse_to_neutral():
    rewards = torch.full((2, 4), 3.0)
    r = batch_zscore_to_r(rewards, adv_clip_max=5.0)
    assert torch.allclose(r, torch.full_like(r, 0.5), atol=1e-6)
