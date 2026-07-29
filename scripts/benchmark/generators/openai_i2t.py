#!/usr/bin/env python3
"""Stub: closed/open figure-to-text caption generation."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure-to-text generator stub")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4o")
    args = parser.parse_args()
    out = ROOT / "outputs" / "benchmark_eval200" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[stub] I2T run {args.run_id} provider={args.provider} model={args.model}\n"
        f"  manifest={args.manifest}\n"
        f"  output={out}\n"
        "Implement VLM caption generation, BERTScore, and Judge*-CC evaluation."
    )


if __name__ == "__main__":
    main()
