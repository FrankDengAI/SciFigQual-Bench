#!/usr/bin/env python3
"""Run Sidecar Judge (S5) on a fixed-size eval1200 subset."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_SCORING = (
    ROOT
    / "code"
    / "Multidimensional-Assessment-of-Scientific-Paper-Figures-main"
    / "model_scoring"
)
SCORE_SCRIPT = MODEL_SCORING / "score_and_upload.py"
SUBSET_MANIFEST = ROOT / "configs" / "human_eval_subset_1200.jsonl"


def _load_lines(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    run_dir = ROOT / "outputs" / "benchmark_eval1200" / "S5"
    run_dir.mkdir(parents=True, exist_ok=True)

    records = _load_lines(SUBSET_MANIFEST, limit)
    manifest_100 = run_dir / f"manifest_{limit}.jsonl"
    with manifest_100.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in records:
        grouped[(str(rec["venue"]), int(rec["year"]))].append(rec)

    jsonl_path = run_dir / "results.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    source_name = "doubao:doubao-seed-2-0-pro-260215:with_features-eval1200-S5"
    shard_paths: list[Path] = []

    for (venue, year), group in sorted(grouped.items()):
        shard_tag = f"{venue}_{year}"
        shard_manifest = run_dir / f"manifest_{shard_tag}.jsonl"
        with shard_manifest.open("w", encoding="utf-8") as handle:
            for rec in group:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        shard_parquet = run_dir / f"shard_{shard_tag}.parquet"
        cmd = [
            sys.executable,
            str(SCORE_SCRIPT),
            "--batch-mode",
            "paper",
            "--stage",
            "clean",
            "--venue",
            venue,
            "--year",
            str(year),
            "--manifest",
            str(shard_manifest),
            "--output-parquet",
            str(shard_parquet),
            "--output-jsonl",
            str(jsonl_path),
            "--skip-upload",
            "--model-label",
            source_name,
            "--workers",
            "1",
            "--provider",
            "doubao",
            "--model",
            "doubao-seed-2-0-pro-260215",
            "--prompt-mode",
            "with_features",
        ]
        print(f"\n>>> {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(MODEL_SCORING), check=True)
        shard_paths.append(shard_parquet)

    import polars as pl

    frames = [pl.read_parquet(p) for p in shard_paths if p.exists()]
    merged = pl.concat(frames, how="diagonal_relaxed")
    merged_path = run_dir / "results.parquet"
    merged.write_parquet(merged_path)
    meta = {
        "run_id": "S5",
        "n_scored": merged.height,
        "n_target_subset": limit,
        "source_name": source_name,
        "model": "doubao-seed-2-0-pro-260215",
        "protocol": "with_features",
    }
    (run_dir / "subset_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nS5 subset done: {merged.height} rows -> {merged_path}")


if __name__ == "__main__":
    main()
