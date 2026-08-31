from pydantic import BaseModel, Field, ConfigDict


class JudgeScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    musicality: int = Field(ge=1, le=5)
    text_music_alignment: int = Field(ge=1, le=5)
    video_music_alignment: int = Field(ge=1, le=5)
    rationale: str = ""


JUDGE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "musicality",
        "text_music_alignment",
        "video_music_alignment",
    ],
    "properties": {
        "musicality": {"type": "integer", "minimum": 1, "maximum": 5},
        "text_music_alignment": {"type": "integer", "minimum": 1, "maximum": 5},
        "video_music_alignment": {"type": "integer", "minimum": 1, "maximum": 5},
        "rationale": {"type": "string"},
    },
}
