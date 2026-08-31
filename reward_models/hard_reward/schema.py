from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class VerifiableScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tempo: Optional[float] = Field(default=None, ge=0.0, le=1.0,
                                   description="BPM agreement; None when the target has no tempo")
    key: float = Field(ge=0.0, le=1.0, description="key/scale agreement")


VERIFIABLE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tempo", "key"],
    "properties": {
        "tempo": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "key": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

#: axis order used by HardVerifiableConfigurator.score_group
AXES = ("tempo", "key")

#: manifest fields each row must carry for these rewards to fire
REQUIRED_ATTRIBUTES = ("tempo", "key", "scale")
