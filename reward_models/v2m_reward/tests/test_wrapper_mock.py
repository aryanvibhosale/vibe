import torch

from reward_models.v2m_reward.wrapper import JudgeRewardTensor
from reward_models.v2m_reward.inference import JudgeRawScores
import reward_models.v2m_reward.wrapper as jw


class _StubInference:

    def __init__(self, *_, **__):
        self.model_id = "stub"
        self.calls = []  # list of dicts describing each call

    def score_one(self, video_path, audio_waveform, sample_rate, text_caption):
        self.calls.append({
            "kind": "one",
            "video": video_path,
            "audio_len": int(audio_waveform.numel()),
            "text": text_caption,
        })
        return JudgeRawScores(3.0, 4.0, 5.0, parsed_ok=True, raw_text="ok")

    def score_prompt_group(self, video_path, text_caption, audios, sample_rate):
        out = []
        for i, a in enumerate(audios):
            self.calls.append({
                "kind": "group",
                "video": video_path,
                "audio_len": int(a.numel()),
                "text": text_caption,
            })
            # Make the result depend on i so we can verify ordering downstream.
            ok = i != 1  # second one fails parsing
            if ok:
                out.append(JudgeRawScores(1.0 + i, 2.0 + i, 3.0 + i, parsed_ok=True, raw_text="ok"))
            else:
                out.append(JudgeRawScores(3.0, 3.0, 3.0, parsed_ok=False, raw_text="bad"))
        return out


def _make_wrapper(monkeypatch):
    monkeypatch.setattr(jw, "JudgeModelInference", _StubInference)
    return JudgeRewardTensor(model_id="stub", device="cpu", dtype="bfloat16",
                             load_in_4bit=False, n_video_frames=4)


def test_score_tensor_returns_three_axes(monkeypatch):
    j = _make_wrapper(monkeypatch)
    out = j.score_tensor(
        waveform=torch.randn(48000),
        sample_rate=48000,
        video_path="/tmp/x.mp4",
        text="hello",
    )
    assert set(out.keys()) == {"musicality", "text_music_alignment", "video_music_alignment"}
    assert out == {"musicality": 3.0, "text_music_alignment": 4.0, "video_music_alignment": 5.0}


def test_mono_flattening_stereo_CT(monkeypatch):
    j = _make_wrapper(monkeypatch)
    j.score_tensor(
        waveform=torch.randn(2, 48000),  # stereo [C, T]
        sample_rate=48000,
        video_path="/tmp/x.mp4",
        text="t",
    )
    assert j.model.calls[-1]["audio_len"] == 48000  # collapsed to mono


def test_mono_flattening_batched(monkeypatch):
    j = _make_wrapper(monkeypatch)
    j.score_tensor(
        waveform=torch.randn(1, 1, 16000),  # batch of mono
        sample_rate=48000,
        video_path="/tmp/x.mp4",
        text="t",
    )
    assert j.model.calls[-1]["audio_len"] == 16000


def test_score_prompt_group_shape_and_order(monkeypatch):
    j = _make_wrapper(monkeypatch)
    G = 4
    wavs = [torch.randn(48000) for _ in range(G)]
    out = j.score_prompt_group(
        video_path="/tmp/v.mp4", text="cap", waveforms=wavs, sample_rate=48000,
    )
    assert out.shape == (G, 3)
    # Index 1 is the simulated parse failure → falls back to all 3.0
    assert torch.allclose(out[1], torch.tensor([3.0, 3.0, 3.0]))
    # Index 0,2,3 should be (1+i, 2+i, 3+i)
    for i in (0, 2, 3):
        assert out[i, 0].item() == 1.0 + i
        assert out[i, 1].item() == 2.0 + i
        assert out[i, 2].item() == 3.0 + i


def test_stats_track_calls_and_failures(monkeypatch):
    j = _make_wrapper(monkeypatch)
    j.score_prompt_group(
        video_path="/tmp/v.mp4", text="cap",
        waveforms=[torch.randn(48000) for _ in range(4)],
        sample_rate=48000,
    )
    s = j.stats()
    assert s["prompts"] == 1
    assert s["calls"] == 4
    assert s["parse_failures"] == 1
    j.reset_stats()
    s2 = j.stats()
    assert s2 == {"prompts": 0, "calls": 0, "parse_failures": 0}


def test_score_prompt_group_passes_video_once(monkeypatch):
    j = _make_wrapper(monkeypatch)
    j.score_prompt_group(
        video_path="/path/to/video.mp4", text="cap",
        waveforms=[torch.randn(1000) for _ in range(3)],
        sample_rate=48000,
    )
    paths = {c["video"] for c in j.model.calls}
    assert paths == {"/path/to/video.mp4"}
