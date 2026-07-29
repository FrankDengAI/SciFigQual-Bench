"""Public display names for the three SciFigQual-Bench scoring protocols.

Code aliases (`baseline`, `with_features`, `agent`) are stable in parquet paths
and CLI flags; only use this module for UI, docs, and report labels.
"""

from __future__ import annotations

PROTOCOL_IDS = ("baseline", "with_features", "agent")

PROTOCOL_DISPLAY_NAMES: dict[str, str] = {
    "baseline": "Direct Judge",
    "with_features": "Sidecar Judge",
    "agent": "SFQ-Agent",
}

PROTOCOL_SHORT_TAGS: dict[str, str] = {
    "baseline": "D",
    "with_features": "S",
    "agent": "F",
}

CANONICAL_PROTOCOL_ID = "agent"


def display_name(protocol_id: str) -> str:
    return PROTOCOL_DISPLAY_NAMES.get(protocol_id, protocol_id)
