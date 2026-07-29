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

from typing import Annotated

import typer
from runner import run_agent_scoring

app = typer.Typer(add_completion=False, pretty_exceptions_short=True)


@app.command()
def main(
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
    vlm_provider: Annotated[
        str, typer.Option(help="Vision model provider: gemini | claude | deepseek | qwen")
    ] = "gemini",
    vlm_model: Annotated[str, typer.Option(help="Vision provider model id")] = "gemini-2.5-flash",
    llm_provider: Annotated[
        str, typer.Option(help="Text judge provider: gemini | claude | deepseek | qwen")
    ] = "gemini",
    llm_model: Annotated[str, typer.Option(help="Text judge model id")] = "gemini-2.5-flash",
    model_label: Annotated[
        str | None,
        typer.Option(help="Source label written to results.parquet source_name field"),
    ] = None,
    features_parquet: Annotated[
        str | None,
        typer.Option(help="Optional feature parquet; OCR columns are included if present"),
    ] = None,
    context_parquet: Annotated[
        str | None,
        typer.Option(help="Optional local figures_context.parquet"),
    ] = None,
    batch_mode: Annotated[str, typer.Option(help="Batch mode: figure | paper")] = "figure",
    max_figures_per_call: Annotated[
        int,
        typer.Option(help="Maximum target figure images in one paper-batch call; capped at 8"),
    ] = 4,
    max_retries: Annotated[int, typer.Option(help="Retries per agent step")] = 2,
    workers: Annotated[int, typer.Option(help="Parallel paper workers for paper batch mode")] = 2,
    skip_upload: Annotated[bool, typer.Option(help="Do not upload merged results")] = False,
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
    run_agent_scoring(
        venue=venue,
        year=year,
        stage=stage,
        parquet=parquet,
        limit=limit,
        paper_id=paper_id,
        paper_limit=paper_limit,
        vlm_provider=vlm_provider,
        vlm_model=vlm_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        model_label=model_label,
        features_parquet=features_parquet,
        context_parquet=context_parquet,
        batch_mode=batch_mode,
        max_figures_per_call=max_figures_per_call,
        max_retries=max_retries,
        workers=workers,
        skip_upload=skip_upload,
        manifest=manifest,
        output_parquet=output_parquet,
        output_jsonl=output_jsonl,
    )


if __name__ == "__main__":
    app()
