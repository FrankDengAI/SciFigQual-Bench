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

import sys
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

AGENT_DIR = Path(__file__).resolve().parent
MODEL_SCORING_DIR = AGENT_DIR.parent
if str(MODEL_SCORING_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_SCORING_DIR))

from io_utils import (  # noqa: E402
    REPO_ID,
    load_env,
    load_existing_results,
    merge_results,
    upload_results,
)

app = typer.Typer(add_completion=False, pretty_exceptions_short=True)


def _source_rows(
    df: pl.DataFrame,
    source_type: str,
    source_name: str,
    upload_source_name: str | None,
) -> pl.DataFrame:
    required = {"paper_id", "fig_index", "source_type", "source_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise typer.BadParameter(f"Local parquet missing required columns: {missing}")
    rows = df.filter(
        (pl.col("source_type") == source_type) & (pl.col("source_name") == source_name)
    )
    if rows.is_empty():
        raise typer.BadParameter(
            f"No rows found for source_type={source_type!r}, source_name={source_name!r}"
        )
    if upload_source_name:
        rows = rows.with_columns(pl.lit(upload_source_name).alias("source_name"))
    return rows


@app.command()
def main(
    parquet: Annotated[str, typer.Option(help="Local merged parquet produced by agent scoring")],
    venue: Annotated[str, typer.Option(help="Conference name, e.g. ACL")],
    year: Annotated[int, typer.Option(help="Conference year")],
    source_name: Annotated[
        str,
        typer.Option(help="Source name to read from local parquet"),
    ] = "agent:gemini:gemini-2.5-flash+gemini:gemini-2.5-flash:v1",
    source_type: Annotated[
        str, typer.Option(help="Source type to read from local parquet")
    ] = "model",
    upload_source_name: Annotated[
        str | None,
        typer.Option(help="Optional shorter source_name to write to HF"),
    ] = "vlm+llm v1",
    dry_run: Annotated[
        bool, typer.Option(help="Validate and show row counts without upload")
    ] = False,
) -> None:
    load_env()
    import os

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise typer.BadParameter("HF_TOKEN is required.")

    local = pl.read_parquet(parquet)
    fresh = _source_rows(local, source_type, source_name, upload_source_name)
    target_source_name = upload_source_name or source_name
    existing = load_existing_results(venue=venue, year=year, token=token)
    merged = merge_results(existing, fresh)

    typer.echo(f"Local parquet rows: {local.height}")
    typer.echo(f"Rows selected for {source_type}:{source_name}: {fresh.height}")
    if upload_source_name:
        typer.echo(f"Renaming source_name on upload: {source_name} -> {upload_source_name}")
    typer.echo(f"Rows will be written as {source_type}:{target_source_name}")
    typer.echo(f"Remote existing rows: {existing.height}")
    typer.echo(f"Merged rows to upload: {merged.height}")
    typer.echo(f"Target: {REPO_ID}/processed/{venue}/{year}/results.parquet")

    if dry_run:
        typer.echo("Dry run only; skipped upload.")
        return

    upload_results(merged, venue=venue, year=year, token=token)
    typer.echo(f"Uploaded processed/{venue}/{year}/results.parquet to {REPO_ID}")


if __name__ == "__main__":
    app()
