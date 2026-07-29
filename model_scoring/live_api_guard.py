"""Block live vendor API calls in the public release build."""

from __future__ import annotations

STUB_MESSAGE = "Set API keys in .env to run live inference (see README)."


def block_live_inference() -> None:
    raise NotImplementedError(STUB_MESSAGE)
