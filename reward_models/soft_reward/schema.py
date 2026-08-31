from pydantic import BaseModel, ConfigDict, Field


class CMIScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alignment: float = Field(description="text/music correspondence")
    quality: float = Field(description="audio quality, text-independent")


CMI_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["alignment", "quality"],
    "properties": {
        "alignment": {"type": "number"},
        "quality": {"type": "number"},
    },
}

#: axis order used by CMIConfigurator.score_group
AXES = ("alignment", "quality")
