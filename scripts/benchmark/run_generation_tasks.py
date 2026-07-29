#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "pyyaml"]
# ///
"""Run Table 2 generation tasks: T2I, I2T, and image-judge on eval200."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_SCORING = ROOT / "code" / "Multidimensional-Assessment-of-Scientific-Paper-Figures-main" / "model_scoring"
AGENT_SCRIPT = MODEL_SCORING / "agent_scoring" / "run_agent_score.py"
MATRIX_SCRIPT = ROOT / "scripts" / "benchmark" / "run_experiment_matrix.py"
DEFAULT_CONFIG = ROOT / "configs" / "eval200_main.yaml"

BENCH_SCRIPTS = ROOT / "scripts" / "benchmark"
if str(BENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCH_SCRIPTS))

from manifest_utils import load_consolidated, write_manifest_jsonl  # noqa: E402


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _build_t2i_prompt(rec: dict) -> str:
    caption = str(rec.get("caption") or "").strip()
    contexts = rec.get("context_texts") or []
    ctx = " ".join(str(t).strip() for t in contexts[:2] if str(t).strip())
    parts = ["Scientific paper figure.", caption]
    if ctx:
        parts.append(f"Context: {ctx}")
    parts.append("High-quality academic visualization with clear labels and axes.")
    return " ".join(p for p in parts if p)


def _build_figure_to_text_prompt() -> str:
    return (
        "Describe this scientific paper figure in a formal figure caption style. "
        "Include main panels, axes, metrics, and compared groups mentioned in the image."
    )


def _plan_run(run: dict, *, manifest_path: Path, output_root: Path) -> dict:
    run_id = run["id"]
    task = run.get("task", "image_judge")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "id": run_id,
        "task": task,
        "phase": run.get("phase"),
        "output_dir": str(run_dir),
        "status": "planned",
    }

    if task == "image_judge":
        record["executor"] = "run_experiment_matrix_subset"
        record["note"] = f"SFQ-Agent scoring via {AGENT_SCRIPT.name}"
        record["command"] = (
            f"python {MATRIX_SCRIPT} --matrix <subset.yaml> --run-id {run_id} --execute"
        )
        return record

    if task == "text_to_figure":
        samples = load_consolidated(manifest_path)[:5]
        prompts = [{"paper_id": r["paper_id"], "fig_index": r["fig_index"], "prompt": _build_t2i_prompt(r)} for r in samples]
        (run_dir / "sample_prompts.json").write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        gen = run.get("generator", {})
        record.update({
            "generator": gen,
            "category": run.get("category"),
            "pipeline": [
                "1. Build prompt from caption + context_texts",
                "2. Call image generation API (generator)",
                "3. Save generated PNG to run_dir/generated/",
                "4. Score with fixed Judge* (SFQ-Agent from Table 1)",
                "5. Optional: CLIPScore vs gold caption",
            ],
            "command": (
                f"# TODO: implement provider={gen.get('provider')} model={gen.get('model')}\n"
                f"python scripts/benchmark/generators/{gen.get('provider', 'local')}_t2i.py "
                f"--run-id {run_id} --manifest {manifest_path}"
            ),
        })
        return record

    if task == "figure_to_text":
        gen = run.get("generator", {})
        record.update({
            "generator": gen,
            "category": run.get("category"),
            "pipeline": [
                "1. Load published figure PNG",
                "2. Generate caption/description with VLM",
                "3. BERTScore vs gold caption",
                "4. Judge*-CC on (image, generated_caption)",
            ],
            "command": (
                f"python scripts/benchmark/generators/{gen.get('provider', 'openai')}_i2t.py "
                f"--run-id {run_id} --manifest {manifest_path} "
                f'--prompt "{_build_figure_to_text_prompt()}"'
            ),
        })
        return record

    record["error"] = f"Unknown task: {task}"
    return record


def run_generation_plan(
    config_path: Path,
    *,
    run_ids: list[str] | None,
    tasks: list[str] | None,
    dry_run: bool,
    execute: bool,
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest_path = _resolve(config["subset"])
    output_root = _resolve(config.get("output_root", "outputs/benchmark_eval200"))
    judge = config.get("judge", {})

    selected = config.get("runs", [])
    if run_ids:
        selected = [r for r in selected if r["id"] in run_ids]
    if tasks:
        selected = [r for r in selected if r.get("task") in tasks]

    plan = {
        "built_at": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "judge": judge,
        "dry_run": dry_run or not execute,
        "runs": [],
    }

    # Task I runs can delegate to matrix runner on eval200 manifest
    image_judge_runs = [r for r in selected if r.get("task") == "image_judge"]
    if execute and image_judge_runs and not dry_run:
        subset_yaml = output_root / "eval200_image_judge_matrix.yaml"
        _write_image_judge_matrix(image_judge_runs, manifest_path, subset_yaml, output_root)

    for run in selected:
        rec = _plan_run(run, manifest_path=manifest_path, output_root=output_root)
        plan["runs"].append(rec)
        summary_path = Path(rec["output_dir"]) / "summary.json"
        summary_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        if execute and not dry_run and run.get("task") == "image_judge":
            cmd = [
                sys.executable,
                str(MATRIX_SCRIPT),
                "--matrix",
                str(output_root / "eval200_image_judge_matrix.yaml"),
                "--run-id",
                run["id"],
                "--execute",
            ]
            print(">>>", " ".join(cmd))
            subprocess.run(cmd, check=True)
            rec["status"] = "done"
            summary_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    plan_path = output_root / "generation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Generation plan written to {plan_path}")
    return plan


def _write_image_judge_matrix(
    runs: list[dict],
    manifest_path: Path,
    dest: Path,
    output_root: Path,
) -> None:
    try:
        subset_rel = str(manifest_path.relative_to(ROOT))
    except ValueError:
        subset_rel = str(manifest_path)
    payload = {
        "subset": subset_rel,
        "output_root": str(output_root),
        "stage": "clean",
        "batch_mode": "paper",
        "skip_upload": True,
        "dry_run": False,
        "runs": [
            {
                "id": r["id"],
                "phase": r.get("phase", "task1"),
                "protocol": "agent",
                "task": "image_judge",
                "vlm": r["vlm"],
                "llm": r["llm"],
                "api_calls_per_fig": r.get("api_calls_per_fig", 3),
            }
            for r in runs
        ],
    }
    dest.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval200 generation-task experiments")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id", action="append", dest="run_ids")
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        choices=["image_judge", "text_to_figure", "figure_to_text"],
    )
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    run_generation_plan(
        args.config,
        run_ids=args.run_ids,
        tasks=args.tasks,
        dry_run=args.dry_run and not args.execute,
        execute=args.execute,
    )


if __name__ == "__main__":
    main()
