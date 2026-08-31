from typing import Optional

#: No system prompt: CMI-RM has no instruction-following interface.
SYSTEM_PROMPT: Optional[str] = None


def build_conditioning_text(caption: str) -> str:
    return " ".join(str(caption).split())
