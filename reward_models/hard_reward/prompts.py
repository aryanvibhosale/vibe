from typing import Any, Dict, Optional, Tuple

from .inference import _BLANKET_RANGES, _SKIP_TERMS

SYSTEM_PROMPT: Optional[str] = None

#: blanket tempo terms recognized by the reward, e.g. "upbeat" -> (115, 150) BPM
BLANKET_TEMPO_TERMS: Dict[str, Tuple[float, float]] = dict(_BLANKET_RANGES)

#: tempo terms that carry no information; the tempo reward is skipped (None)
SKIP_TEMPO_TERMS = set(_SKIP_TERMS)


def build_targets(attributes: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not attributes:
        return None
    if not all(k in attributes for k in ("tempo", "key", "scale")):
        return None
    return {
        "expected_tempo": attributes["tempo"],
        "expected_key": {"key": attributes["key"], "scale": attributes["scale"]},
    }
