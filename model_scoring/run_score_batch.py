#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub>=0.22",
#   "polars",
#   "python-dotenv",
#   "typer>=0.12.0",
# ]
# ///
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from dotenv import load_dotenv
from provider_registry import validate_provider

REPO_ID = "ccf-team/ccf-paper-figures"
HF_BASE = f"hf://datasets/{REPO_ID}"
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
SCRIPT_PATH = REPO_ROOT / "model_scoring" / "score_and_upload.py"
app = typer.Typer(add_completion=False, pretty_exceptions_short=True)


def _load_index(token: str) -> pl.DataFrame:
    return pl.read_parquet(
        f"{HF_BASE}/processed/index.parquet",
        storage_options={"token": token},
    )


@app.command()
def main(
    provider: Annotated[str, typer.Option(help="Model provider: gemini | claude | deepseek | qwen")] = "gemini",
    venue: Annotated[str, typer.Option(help="Conference name")] = "ACL",
    years: Annotated[
        list[int] | None, typer.Option("--year", help="Optional repeatable year filter")
    ] = None,
    stage: Annotated[str, typer.Option(help="clean | raw")] = "clean",
    limit_per_year: Annotated[int | None, typer.Option(help="Optional cap per year")] = None,
    model: Annotated[str, typer.Option(help="Provider model id")] = "gemini-2.5-flash",
    model_label: Annotated[
        str | None,
        typer.Option(help="Source label written to results source_name"),
    ] = None,
    prompt_mode: Annotated[
        str, typer.Option(help="Prompt mode: baseline | with_features")
    ] = "baseline",
    features_parquet: Annotated[
        str | None, typer.Option(help="Optional local features parquet")
    ] = None,
    context_parquet: Annotated[
        str | None, typer.Option(help="Optional local figures_context.parquet")
    ] = None,
    batch_mode: Annotated[str, typer.Option(help="Batch mode: figure | paper")] = "figure",
    max_figures_per_call: Annotated[
        int, typer.Option(help="Maximum target figure images in one paper-batch call")
    ] = 5,
    max_retries: Annotated[int, typer.Option(help="Retries per figure")] = 2,
    skip_upload: Annotated[bool, typer.Option(help="Dry-run mode; local outputs only")] = False,
) -> None:
    load_dotenv(ENV_PATH)
    validate_provider(provider)
    if prompt_mode not in {"baseline", "with_features"}:
        raise typer.BadParameter("--prompt-mode must be baseline or with_features.")
    if batch_mode not in {"figure", "paper"}:
        raise typer.BadParameter("--batch-mode must be figure or paper.")
    if max_figures_per_call < 1 or max_figures_per_call > 8:
        raise typer.BadParameter("--max-figures-per-call must be between 1 and 8.")
    if max_retries < 1:
        raise typer.BadParameter("--max-retries must be >= 1.")
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise typer.BadParameter("HF_TOKEN is required.")

    index = _load_index(token)
    target_years = (
        sorted(set(years))
        if years is not None and years
        else sorted(index.filter(pl.col("venue") == venue)["year"].unique().to_list())
    )
    if not target_years:
        raise typer.BadParameter(f"No years found for venue={venue}.")

    failed: list[int] = []
    for year in target_years:
        cmd = [
            "uv",
            "run",
            "--script",
            str(SCRIPT_PATH),
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
            "--batch-mode",
            batch_mode,
            "--max-figures-per-call",
            str(max_figures_per_call),
            "--max-retries",
            str(max_retries),
        ]
        if model_label:
            cmd += ["--model-label", model_label]
        if features_parquet:
            cmd += ["--features-parquet", features_parquet]
        if context_parquet:
            cmd += ["--context-parquet", context_parquet]
        if limit_per_year is not None:
            cmd += ["--limit", str(limit_per_year)]
        if skip_upload:
            cmd += ["--skip-upload"]

        typer.echo(f"\n=== {provider} {venue} {year} ===")
        result = subprocess.run(cmd, cwd=str(REPO_ROOT))
        if result.returncode != 0:
            failed.append(year)

    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
