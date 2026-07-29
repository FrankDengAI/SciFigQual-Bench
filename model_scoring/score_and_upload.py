#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic>=0.49.0",
#   "google-genai>=1.0.0",
#   "huggingface-hub>=0.22",
#   "polars",
#   "pydantic>=2.0",
#   "python-dotenv",
#   "typer>=0.12.0",
#   "openai>=1.0.0",
# ]
# ///
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from typing import Annotated

import typer
from runner import run_scoring

app = typer.Typer(add_completion=False, pretty_exceptions_short=True)


@app.command()
def main(
    provider: Annotated[str, typer.Option(help="Model provider: gemini | claude | deepseek | qwen")] = "gemini",
    venue: Annotated[str, typer.Option(help="Conference name, e.g. ACL")] = "",
    year: Annotated[int, typer.Option(help="Conference year")] = 0,
    stage: Annotated[str, typer.Option(help="clean | raw")] = "clean",
    parquet: Annotated[str | None, typer.Option(help="Local figures parquet path")] = None,
    limit: Annotated[int | None, typer.Option(help="Max rows to score")] = None,
    paper_id: Annotated[str | None, typer.Option(help="Score only one paper")] = None,
    paper_limit: Annotated[
        int | None,
        typer.Option(help="Score all figures from the first N papers by paper_id sort order"),
    ] = None,
    model: Annotated[str, typer.Option(help="Provider model id")] = "gemini-2.5-flash",
    model_label: Annotated[
        str | None,
        typer.Option(help="Source label written to results.parquet source_name field"),
    ] = None,
    prompt_mode: Annotated[
        str, typer.Option(help="Prompt mode: baseline | with_features")
    ] = "baseline",
    features_parquet: Annotated[
        str | None, typer.Option(help="Feature parquet used by with_features mode")
    ] = None,
    context_parquet: Annotated[
        str | None, typer.Option(help="Optional local figures_context.parquet")
    ] = None,
    batch_mode: Annotated[str, typer.Option(help="Batch mode: figure | paper")] = "figure",
    max_figures_per_call: Annotated[
        int,
        typer.Option(help="Maximum target figure images in one paper-batch call; capped at 8"),
    ] = 5,
    max_retries: Annotated[int, typer.Option(help="Retries per figure")] = 2,
    skip_upload: Annotated[bool, typer.Option(help="Do not upload merged results")] = False,
    workers: Annotated[int, typer.Option(help="Parallel API workers (default 4)")] = 4,
    manifest: Annotated[
        str | None,
        typer.Option(help="JSONL manifest of (paper_id, fig_index) to score"),
    ] = None,
    output_parquet: Annotated[
        str | None,
        typer.Option(help="Write merged results to this parquet path"),
    ] = None,
    output_jsonl: Annotated[
        str | None,
        typer.Option(help="Append each scored figure as one JSONL line"),
    ] = None,
) -> None:
    run_scoring(
        provider=provider,
        venue=venue,
        year=year,
        stage=stage,
        parquet=parquet,
        limit=limit,
        paper_id=paper_id,
        paper_limit=paper_limit,
        model=model,
        model_label=model_label,
        prompt_mode=prompt_mode,
        features_parquet=features_parquet,
        context_parquet=context_parquet,
        batch_mode=batch_mode,
        max_figures_per_call=max_figures_per_call,
        max_retries=max_retries,
        skip_upload=skip_upload,
        workers=workers,
        manifest=manifest,
        output_parquet=output_parquet,
        output_jsonl=output_jsonl,
    )


if __name__ == "__main__":
    app()
