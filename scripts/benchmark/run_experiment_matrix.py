#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "pyyaml"]
# ///
"""Drive eval experiment matrix: Direct / Sidecar / SFQ-Agent across model backends."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import yaml

from manifest_utils import load_consolidated, write_manifest_jsonl

ROOT = Path(__file__).resolve().parents[2]
MODEL_SCORING = (
    ROOT
    / "code"
    / "Multidimensional-Assessment-of-Scientific-Paper-Figures-main"
    / "model_scoring"
)
SCORE_SCRIPT = MODEL_SCORING / "score_and_upload.py"
AGENT_SCRIPT = MODEL_SCORING / "agent_scoring" / "run_agent_score.py"
DEFAULT_MATRIX = ROOT / "configs" / "eval1200_ablation.yaml"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _shard_manifest(manifest_path: Path, venue: str, year: int, out_path: Path) -> int:
    records = [r for r in load_consolidated(manifest_path) if r.get("venue") == venue and int(r["year"]) == year]
    write_manifest_jsonl(records, out_path)
    return len(records)


def _source_name(run: dict, run_id: str, eval_tag: str = "eval1200") -> str:
    protocol = run["protocol"]
    if protocol == "agent":
        vlm = run["vlm"]
        llm = run["llm"]
        return (
            f"{vlm['provider']}:{vlm['model']}+{llm['provider']}:{llm['model']}:"
            f"agent-{eval_tag}-{run_id}"
        )
    provider = run["provider"]
    model = run["model"]
    if protocol == "with_features":
        return f"{provider}:{model}:with_features-{eval_tag}-{run_id}"
    return f"{provider}:{model}:baseline-{eval_tag}-{run_id}"


def _build_command(
    run: dict,
    *,
    venue: str,
    year: int,
    shard_manifest: Path,
    output_parquet: Path,
    output_jsonl: Path,
    stage: str,
    batch_mode: str,
) -> list[str]:
    run_id = run["id"]
    eval_tag = run.get("eval_tag", "eval1200")
    source_name = _source_name(run, run_id, eval_tag)
    protocol = run["protocol"]
    common = [
        "--batch-mode",
        batch_mode,
        "--stage",
        stage,
        "--venue",
        venue,
        "--year",
        str(year),
        "--manifest",
        str(shard_manifest),
        "--output-parquet",
        str(output_parquet),
        "--output-jsonl",
        str(output_jsonl),
        "--skip-upload",
        "--model-label",
        source_name,
        "--workers",
        "1",
    ]

    if protocol == "agent":
        vlm = run["vlm"]
        llm = run["llm"]
        return [
            sys.executable,
            str(AGENT_SCRIPT),
            *common,
            "--vlm-provider",
            vlm["provider"],
            "--vlm-model",
            vlm["model"],
            "--llm-provider",
            llm["provider"],
            "--llm-model",
            llm["model"],
        ]

    return [
        sys.executable,
        str(SCORE_SCRIPT),
        *common,
        "--provider",
        run["provider"],
        "--model",
        run["model"],
        "--prompt-mode",
        protocol,
    ]


def _merge_shards(shard_paths: list[Path], dest: Path) -> None:
    frames = [pl.read_parquet(p) for p in shard_paths if p.exists() and p.stat().st_size > 0]
    if not frames:
        raise RuntimeError(f"No shard parquet files to merge for {dest}")
    merged = pl.concat(frames, how="diagonal_relaxed")
    dest.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(dest)


def run_matrix(
    matrix_path: Path,
    *,
    run_ids: list[str] | None,
    phases: list[str] | None,
    dry_run: bool | None,
) -> dict:
    config = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    manifest_path = _resolve(config["subset"])
    output_root = _resolve(config.get("output_root", "outputs/benchmark_eval1200"))
    eval_tag = Path(str(config.get("subset", ""))).stem.replace("human_eval_subset_", "eval")
    if not eval_tag.startswith("eval"):
        eval_tag = "eval1200"
    stage = config.get("stage", "clean")
    batch_mode = config.get("batch_mode", "paper")
    matrix_dry_run = bool(config.get("dry_run", False))
    effective_dry_run = matrix_dry_run if dry_run is None else dry_run

    manifest_df = pl.DataFrame(load_consolidated(manifest_path))
    shards = (
        manifest_df.group_by("venue", "year")
        .agg(pl.len().alias("n_figs"))
        .sort(["venue", "year"])
    )

    selected_runs = config["runs"]
    if run_ids:
        selected_runs = [r for r in selected_runs if r["id"] in run_ids]
    if phases:
        selected_runs = [r for r in selected_runs if r.get("phase") in phases]

    plan: dict = {
        "built_at": datetime.now(UTC).isoformat(),
        "matrix": str(matrix_path),
        "manifest": str(manifest_path),
        "dry_run": effective_dry_run,
        "runs": [],
    }

    for run in selected_runs:
        run_id = run["id"]
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = run_dir / "results.jsonl"
        if jsonl_path.exists():
            jsonl_path.unlink()
        shard_outputs: list[Path] = []
        shard_cmds: list[list[str]] = []

        for row in shards.iter_rows(named=True):
            venue = str(row["venue"])
            year = int(row["year"])
            shard_tag = f"{venue}_{year}"
            shard_manifest = run_dir / f"manifest_{shard_tag}.jsonl"
            n_figs = _shard_manifest(manifest_path, venue, year, shard_manifest)
            if n_figs == 0:
                continue
            shard_parquet = run_dir / f"shard_{shard_tag}.parquet"
            cmd = _build_command(
                run,
                venue=venue,
                year=year,
                shard_manifest=shard_manifest,
                output_parquet=shard_parquet,
                output_jsonl=jsonl_path,
                stage=stage,
                batch_mode=batch_mode,
            )
            shard_cmds.append(cmd)
            shard_outputs.append(shard_parquet)

        merged_path = run_dir / "results.parquet"
        run_record = {
            "id": run_id,
            "phase": run.get("phase"),
            "protocol": run["protocol"],
            "source_name": _source_name(run, run_id, eval_tag),
            "api_calls_per_fig": run.get("api_calls_per_fig"),
            "output": str(merged_path),
            "output_jsonl": str(jsonl_path),
            "shard_commands": [" ".join(cmd) for cmd in shard_cmds],
        }
        plan["runs"].append(run_record)

        if effective_dry_run:
            continue

        for cmd in shard_cmds:
            print(f"\n>>> {' '.join(cmd)}")
            subprocess.run(cmd, cwd=str(MODEL_SCORING), check=True)
        _merge_shards(shard_outputs, merged_path)
        print(f"Merged {len(shard_outputs)} shards -> {merged_path}")

    plan_path = output_root / "experiment_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"\nExperiment plan written to {plan_path}")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval experiment matrix")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--run-id", action="append", dest="run_ids", help="Limit to run id(s)")
    parser.add_argument("--phase", action="append", dest="phases", help="Limit to phase A/B/C")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually invoke scoring scripts (default: dry-run plan only)",
    )
    args = parser.parse_args()
    run_matrix(
        args.matrix,
        run_ids=args.run_ids,
        phases=args.phases,
        dry_run=not args.execute,
    )


if __name__ == "__main__":
    main()
