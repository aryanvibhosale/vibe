from reward_models.v2m_reward.inference import JudgeModelInference


def test_parse_group_accepts_exact_length_array():
    j = JudgeModelInference.__new__(JudgeModelInference)
    raw = """
    [
      {"musicality": 3, "text_music_alignment": 2, "video_music_alignment": 1},
      {"musicality": 4, "text_music_alignment": 5, "video_music_alignment": 3}
    ]
    """
    out = j._parse_group(raw, expected_len=2)
    assert out is not None
    assert len(out) == 2
    assert out[0].musicality == 3.0
    assert out[1].text_music_alignment == 5.0
    assert all(r.parsed_ok for r in out)


def test_parse_group_rejects_wrong_length():
    j = JudgeModelInference.__new__(JudgeModelInference)
    raw = '[{"musicality": 3, "text_music_alignment": 2, "video_music_alignment": 1}]'
    assert j._parse_group(raw, expected_len=2) is None


def test_parse_group_rejects_bad_score():
    j = JudgeModelInference.__new__(JudgeModelInference)
    raw = '[{"musicality": 6, "text_music_alignment": 2, "video_music_alignment": 1}]'
    assert j._parse_group(raw, expected_len=1) is None


class _ChunkProbe(JudgeModelInference):
    def __init__(self):
        self.group_chunk_size = 3
        self.calls = []

    def _score_prompt_group_single_turn_multi_candidate(
        self,
        video_path,
        text_caption,
        audios,
        sample_rate,
    ):
        self.calls.append(len(audios))
        return [
            type("Score", (), {
                "musicality": float(i),
                "text_music_alignment": 2.0,
                "video_music_alignment": 1.0,
                "parsed_ok": True,
                "raw_text": "",
            })()
            for i, _ in enumerate(audios)
        ]


def test_chunked_group_scoring_splits_expected_sizes():
    j = _ChunkProbe()
    out = j._score_prompt_group_chunked_single_turn_multi_candidate(
        video_path="v.mp4",
        text_caption="caption",
        audios=list(range(8)),
        sample_rate=48000,
    )
    assert j.calls == [3, 3, 2]
    assert len(out) == 8
