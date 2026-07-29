#!/usr/bin/env python3
"""Stub: local/open-source text-to-figure generation (Flux, SDXL, etc.)."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Local T2I generator stub")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="flux.1-dev")
    args = parser.parse_args()
    out = ROOT / "outputs" / "benchmark_eval200" / args.run_id / "generated"
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[stub] T2I run {args.run_id} model={args.model}\n"
        f"  manifest={args.manifest}\n"
        f"  output={out}\n"
        "Implement API/local diffusion call, then score with fixed Judge*."
    )


if __name__ == "__main__":
    main()
