#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "polars",
#   "python-dotenv",
#   "typer>=0.12.0",
# ]
# ///
from __future__ import annotations

import itertools
import json
import re
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, cast

import polars as pl
import typer

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORE_SCRIPT = REPO_ROOT / "model_scoring" / "score_and_upload.py"
AGENT_SCORE_SCRIPT = REPO_ROOT / "model_scoring" / "agent_scoring" / "run_agent_score.py"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "model_scoring" / "reproducibility"
ScoringMode = Literal["baseline", "with_features", "agent"]

DIMENSIONS = [
    "visual_clarity",
    "structure_layout",
    "caption_consistency",
    "context_consistency",
    "misleading_risk",
    "overall_score",
]

app = typer.Typer(add_completion=False, pretty_exceptions_short=True)


def _safe_label(label: str) -> str:
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "-", label).strip().strip(".")
    return cleaned or "model-output"


def _run_once(
    *,
    scoring_mode: ScoringMode,
    provider: str,
    venue: str,
    year: int,
    stage: str,
    model: str,
    vlm_provider: str,
    vlm_model: str,
    llm_provider: str,
    llm_model: str,
    parquet: str | None,
    features_parquet: str | None,
    context_parquet: str | None,
    batch_mode: str,
    max_figures_per_call: int | None,
    max_retries: int,
    workers: int,
    base_model_label: str,
    paper_limit: int | None,
    run_index: int,
) -> tuple[str, Path]:
    run_label = f"{base_model_label}__run{run_index}"
    safe_run_label = _safe_label(run_label)
    if scoring_mode == "agent":
        output_providers = ["agent"]
    else:
        # score_and_upload.py writes batch executions under paper_batch even when
        # --batch-mode is "figure"; keep the provider path as a fallback for
        # older outputs and non-batch layouts.
        output_providers = ["paper_batch", provider]
    expected_paths = [
        REPO_ROOT
        / "outputs"
        / "model_scoring"
        / output_provider
        / venue
        / str(year)
        / f"{safe_run_label}.parquet"
        for output_provider in output_providers
    ]
    for existing_path in expected_paths:
        if existing_path.exists():
            typer.echo(f"Reusing existing output parquet: {existing_path}")
            return run_label, existing_path

    if scoring_mode == "agent":
        cmd = [
            "uv",
            "run",
            "--script",
            str(AGENT_SCORE_SCRIPT),
            "--venue",
            venue,
            "--year",
            str(year),
            "--stage",
            stage,
            "--vlm-provider",
            vlm_provider,
            "--vlm-model",
            vlm_model,
            "--llm-provider",
            llm_provider,
            "--llm-model",
            llm_model,
            "--model-label",
            run_label,
            "--batch-mode",
            batch_mode,
            "--max-retries",
            str(max_retries),
            "--workers",
            str(workers),
            "--skip-upload",
        ]
        if paper_limit is not None:
            cmd += ["--paper-limit", str(paper_limit)]
        if parquet:
            cmd += ["--parquet", parquet]
        if max_figures_per_call is not None:
            cmd += ["--max-figures-per-call", str(max_figures_per_call)]
        if features_parquet:
            cmd += ["--features-parquet", features_parquet]
        if context_parquet:
            cmd += ["--context-parquet", context_parquet]
    else:
        prompt_mode = "baseline" if scoring_mode == "baseline" else "with_features"
        cmd = [
            "uv",
            "run",
            "--script",
            str(SCORE_SCRIPT),
            "--venue",
            venue,
            "--year",
            str(year),
            "--provider",
            provider,
            "--stage",
            stage,
            "--model",
            model,
            "--prompt-mode",
            prompt_mode,
            "--model-label",
            run_label,
            "--batch-mode",
            batch_mode,
            "--max-retries",
            str(max_retries),
            "--workers",
            str(workers),
            "--skip-upload",
        ]
        if paper_limit is not None:
            cmd += ["--paper-limit", str(paper_limit)]
        if parquet:
            cmd += ["--parquet", parquet]
        if max_figures_per_call is not None:
            cmd += ["--max-figures-per-call", str(max_figures_per_call)]
        if features_parquet:
            cmd += ["--features-parquet", features_parquet]
        if context_parquet:
            cmd += ["--context-parquet", context_parquet]

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)
    for parquet_path in expected_paths:
        if parquet_path.exists():
            return run_label, parquet_path
    expected = "\n".join(str(path) for path in expected_paths)
    raise FileNotFoundError(f"Expected output parquet not found in:\n{expected}")


def _load_run(path: Path, run_label: str) -> pl.DataFrame:
    return pl.read_parquet(path).filter(
        (pl.col("source_type") == "model") & (pl.col("source_name") == run_label)
    )


def _pairwise_metrics(run_tables: list[tuple[str, pl.DataFrame]]) -> tuple[list[dict], list[dict]]:
    per_pair: list[dict] = []
    per_dimension_summary: list[dict] = []
    dim_collect: dict[str, list[float]] = {dim: [] for dim in DIMENSIONS}

    for (label_a, df_a), (label_b, df_b) in itertools.combinations(run_tables, 2):
        joined = df_a.join(
            df_b,
            on=["paper_id", "fig_index"],
            how="inner",
            suffix="_b",
        )
        if joined.is_empty():
            continue
        for dim in DIMENSIONS:
            diff_expr = (pl.col(dim) - pl.col(f"{dim}_b")).abs()
            metrics = joined.select(
                diff_expr.mean().alias("mad"),
                diff_expr.max().alias("maxd"),
                diff_expr.drop_nulls().count().alias("count"),
            ).row(0, named=True)
            if metrics["count"] == 0:
                continue
            mean_abs_diff = float(metrics["mad"])
            max_abs_diff = float(metrics["maxd"])
            count = int(joined.height)
            per_pair.append(
                {
                    "run_a": label_a,
                    "run_b": label_b,
                    "dimension": dim,
                    "mean_abs_diff": round(mean_abs_diff, 4),
                    "max_abs_diff": round(max_abs_diff, 4),
                    "n_common_rows": count,
                }
            )
            dim_collect[dim].append(mean_abs_diff)

    for dim, values in dim_collect.items():
        if not values:
            continue
        per_dimension_summary.append(
            {
                "dimension": dim,
                "pair_count": len(values),
                "mean_pairwise_abs_diff": round(sum(values) / len(values), 4),
                "max_pairwise_abs_diff": round(max(values), 4),
            }
        )

    return per_pair, per_dimension_summary


def _format_run_scores(row: dict) -> str:
    return (
        f"VC `{row['visual_clarity']}`"
        f", SL `{row['structure_layout']}`"
        f", CC `{row['caption_consistency']}`"
        f", CTX `{row['context_consistency']}`"
        f", MR `{row['misleading_risk']}`"
        f", Overall `{row['overall_score']}`"
    )


def _variation_line(rows: list[dict]) -> str:
    labels = [
        ("visual_clarity", "VC"),
        ("structure_layout", "SL"),
        ("caption_consistency", "CC"),
        ("context_consistency", "CTX"),
        ("misleading_risk", "MR"),
        ("overall_score", "Overall"),
    ]
    parts: list[str] = []
    for key, label in labels:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            continue
        low = min(values)
        high = max(values)
        if low != high:
            parts.append(f"{label} `{low}-{high}`")
    return ", ".join(parts) if parts else "No variation."


def _render_report(
    *,
    scoring_mode: ScoringMode,
    venue: str,
    year: int,
    provider: str,
    model: str,
    vlm_provider: str,
    vlm_model: str,
    llm_provider: str,
    llm_model: str,
    parquet: str | None,
    batch_mode: str,
    max_figures_per_call: int | None,
    paper_limit: int | None,
    repeats: int,
    run_tables: list[tuple[str, pl.DataFrame]],
    per_dimension_summary: list[dict],
) -> str:
    if scoring_mode == "agent":
        model_line = (
            f"Mode: `agent`  VLM: `{vlm_provider}/{vlm_model}`  LLM: `{llm_provider}/{llm_model}`"
        )
    else:
        model_line = f"Mode: `{scoring_mode}`  Provider: `{provider}`  Model: `{model}`"

    sections = [
        "# Reproducibility Check",
        "",
        f"Venue: `{venue}`  Year: `{year}`",
        model_line,
        (
            f"Batch mode: `{batch_mode}`  Max figures per call: "
            f"`{max_figures_per_call or 'script default'}`"
        ),
        (
            f"Paper limit: `{paper_limit if paper_limit is not None else 'none'}`  "
            f"Repeats: `{repeats}`"
        ),
        "",
        "## Summary",
    ]

    for row in per_dimension_summary:
        sections.append(
            f"- `{row['dimension']}`: "
            f"mean pairwise abs diff `{row['mean_pairwise_abs_diff']:.4f}`, "
            f"max `{row['max_pairwise_abs_diff']:.4f}`"
        )

    sections.extend(["", "## Per Figure"])

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for run_idx, (_, df) in enumerate(run_tables, start=1):
        for row in df.sort(["paper_id", "fig_index"]).to_dicts():
            row["run_index"] = run_idx
            grouped[(row["paper_id"], int(row["fig_index"]))].append(row)

    for paper_id, fig_index in sorted(grouped):
        rows = sorted(grouped[(paper_id, fig_index)], key=lambda item: item["run_index"])
        sections.extend(
            [
                "",
                f"### `{paper_id}` / Figure `{fig_index}`",
            ]
        )
        for row in rows:
            sections.append(f"- Run {row['run_index']}: {_format_run_scores(row)}")
        sections.append(f"- Variation: {_variation_line(rows)}")

    sections.append("")
    return "\n".join(sections)


@app.command()
def main(
    scoring_mode: Annotated[
        str, typer.Option(help="Scoring mode: baseline | with_features | agent")
    ] = "baseline",
    provider: Annotated[str, typer.Option(help="Model provider: gemini | claude")] = "gemini",
    venue: Annotated[str, typer.Option(help="Conference name, e.g. ACL")] = "ACL",
    year: Annotated[int, typer.Option(help="Conference year")] = 2025,
    stage: Annotated[str, typer.Option(help="clean | raw")] = "clean",
    model: Annotated[str, typer.Option(help="Provider model id")] = "gemini-2.5-flash",
    vlm_provider: Annotated[
        str, typer.Option(help="Agent vision model provider: gemini | claude")
    ] = "gemini",
    vlm_model: Annotated[
        str, typer.Option(help="Agent vision provider model id")
    ] = "gemini-2.5-flash",
    llm_provider: Annotated[
        str, typer.Option(help="Agent text judge provider: gemini | claude")
    ] = "gemini",
    llm_model: Annotated[str, typer.Option(help="Agent text judge model id")] = (
        "gemini-2.5-flash"
    ),
    features_parquet: Annotated[
        str | None, typer.Option(help="Optional local features parquet")
    ] = None,
    parquet: Annotated[
        str | None,
        typer.Option(help="Optional local figures parquet; useful for fixed samples"),
    ] = None,
    context_parquet: Annotated[
        str | None, typer.Option(help="Optional local figures_context.parquet")
    ] = None,
    batch_mode: Annotated[
        str, typer.Option(help="Batch mode passed to scoring scripts: figure | paper")
    ] = "figure",
    max_figures_per_call: Annotated[
        int | None,
        typer.Option(
            help=(
                "Maximum target figure images in one paper-batch call; "
                "omit to use scoring-script default"
            )
        ),
    ] = None,
    max_retries: Annotated[int, typer.Option(help="Retries per figure/chunk/agent step")] = 2,
    workers: Annotated[int, typer.Option(help="Parallel workers passed to scoring scripts")] = 2,
    paper_limit: Annotated[
        int | None,
        typer.Option(
            help=("Number of first papers to score. Omit when --parquet is already a fixed sample.")
        ),
    ] = 5,
    repeats: Annotated[int, typer.Option(help="Number of repeated runs")] = 3,
    output_root: Annotated[
        Path,
        typer.Option(help="Directory for reproducibility summary outputs"),
    ] = OUTPUT_ROOT,
    model_label: Annotated[
        str, typer.Option(help="Base model label prefix")
    ] = "gemini:flash-repro-v1",
) -> None:
    if scoring_mode not in {"baseline", "with_features", "agent"}:
        raise typer.BadParameter("--scoring-mode must be baseline, with_features, or agent.")
    mode = cast(ScoringMode, scoring_mode)
    if provider not in {"gemini", "claude", "deepseek", "qwen"}:
        raise typer.BadParameter("--provider must be gemini, claude, deepseek, or qwen.")
    for option_name, option_value in [
        ("vlm-provider", vlm_provider),
        ("llm-provider", llm_provider),
    ]:
        if option_value not in {"gemini", "claude", "deepseek", "qwen"}:
            raise typer.BadParameter(f"--{option_name} must be gemini, claude, deepseek, or qwen.")
    if repeats < 2:
        raise typer.BadParameter("--repeats must be at least 2.")
    if batch_mode not in {"figure", "paper"}:
        raise typer.BadParameter("--batch-mode must be figure or paper.")
    if max_figures_per_call is not None and (max_figures_per_call < 1 or max_figures_per_call > 8):
        raise typer.BadParameter("--max-figures-per-call must be between 1 and 8.")
    if max_retries < 1:
        raise typer.BadParameter("--max-retries must be >= 1.")
    if workers < 1:
        raise typer.BadParameter("--workers must be >= 1.")
    if paper_limit is not None and paper_limit <= 0:
        paper_limit = None

    run_tables: list[tuple[str, pl.DataFrame]] = []
    output_dir = output_root / venue / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(1, repeats + 1):
        typer.echo(f"\n=== reproducibility run {idx}/{repeats} ===")
        run_label, parquet_path = _run_once(
            scoring_mode=mode,
            provider=provider,
            venue=venue,
            year=year,
            stage=stage,
            model=model,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            llm_provider=llm_provider,
            llm_model=llm_model,
            parquet=parquet,
            features_parquet=features_parquet,
            context_parquet=context_parquet,
            batch_mode=batch_mode,
            max_figures_per_call=max_figures_per_call,
            max_retries=max_retries,
            workers=workers,
            base_model_label=model_label,
            paper_limit=paper_limit,
            run_index=idx,
        )
        run_tables.append((run_label, _load_run(parquet_path, run_label)))

    per_pair, per_dimension_summary = _pairwise_metrics(run_tables)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_path = output_dir / f"{_safe_label(model_label)}.{timestamp}.summary.json"
    pairwise_path = output_dir / f"{_safe_label(model_label)}.{timestamp}.pairwise.parquet"
    report_path = output_dir / f"{_safe_label(model_label)}.{timestamp}.report.md"

    summary_payload = {
        "scoring_mode": mode,
        "venue": venue,
        "year": year,
        "stage": stage,
        "provider": provider,
        "model": model,
        "vlm_provider": vlm_provider,
        "vlm_model": vlm_model,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "parquet": parquet,
        "features_parquet": features_parquet,
        "context_parquet": context_parquet,
        "batch_mode": batch_mode,
        "max_figures_per_call": max_figures_per_call,
        "max_retries": max_retries,
        "workers": workers,
        "paper_limit": paper_limit,
        "repeats": repeats,
        "run_labels": [label for label, _ in run_tables],
        "dimension_summary": per_dimension_summary,
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pl.DataFrame(per_pair).write_parquet(pairwise_path)
    report_path.write_text(
        _render_report(
            scoring_mode=mode,
            venue=venue,
            year=year,
            provider=provider,
            model=model,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            llm_provider=llm_provider,
            llm_model=llm_model,
            parquet=parquet,
            batch_mode=batch_mode,
            max_figures_per_call=max_figures_per_call,
            paper_limit=paper_limit,
            repeats=repeats,
            run_tables=run_tables,
            per_dimension_summary=per_dimension_summary,
        ),
        encoding="utf-8",
    )

    typer.echo(f"Summary written to {summary_path}")
    typer.echo(f"Pairwise details written to {pairwise_path}")
    typer.echo(f"Report written to {report_path}")


if __name__ == "__main__":
    app()
