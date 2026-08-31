import pytest
from pydantic import ValidationError

from reward_models.v2m_reward.schema import JudgeScores


def test_clean_parse():
    s = JudgeScores(musicality=4, text_music_alignment=3, video_music_alignment=5)
    assert s.musicality == 4
    assert s.rationale == ""


def test_with_rationale():
    s = JudgeScores(musicality=2, text_music_alignment=2, video_music_alignment=2,
                    rationale="muddy mix, off-tempo")
    assert s.rationale.startswith("muddy")


def test_out_of_range_low():
    with pytest.raises(ValidationError):
        JudgeScores(musicality=0, text_music_alignment=3, video_music_alignment=3)


def test_out_of_range_high():
    with pytest.raises(ValidationError):
        JudgeScores(musicality=3, text_music_alignment=6, video_music_alignment=3)


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        JudgeScores(musicality=3, text_music_alignment=3, video_music_alignment=3,
                    bonus=42)


def test_missing_field_rejected():
    with pytest.raises(ValidationError):
        JudgeScores(musicality=3, text_music_alignment=3)
