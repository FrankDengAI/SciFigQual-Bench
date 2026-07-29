#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "huggingface-hub>=0.22",
#   "polars",
#   "pyarrow",
#   "python-dotenv",
#   "typer>=0.12.0",
# ]
# ///
"""Upload a figure-level feature parquet to the HF dataset."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cs64.data import DEFAULT_HF_REPO_ID, load_project_env

app = typer.Typer(add_completion=False, pretty_exceptions_short=True)


def _validate_feature_table(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise typer.BadParameter(f"Feature parquet not found: {path}")

    df = pl.read_parquet(path)
    missing = sorted({"paper_id", "fig_index"} - set(df.columns))
    if missing:
        raise typer.BadParameter(f"Feature parquet missing required columns: {missing}")
    return df


@app.command()
def main(
    input: Annotated[Path, typer.Option(help="Local figures_features.parquet path")],
    venue: Annotated[str, typer.Option(help="Conference name, e.g. ACL")],
    year: Annotated[int, typer.Option(help="Conference year")],
    repo_id: Annotated[str, typer.Option(help="HF dataset repo id")] = DEFAULT_HF_REPO_ID,
    dry_run: Annotated[bool, typer.Option(help="Validate only; do not upload")] = False,
) -> None:
    load_project_env()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise typer.BadParameter("HF_TOKEN is required.")

    df = _validate_feature_table(input)
    repo_path = f"processed/{venue}/{year}/features/figures_features.parquet"

    typer.echo(f"Feature rows: {df.height}")
    typer.echo(f"Local path: {input}")
    typer.echo(f"HF target: {repo_id}/{repo_path}")

    if dry_run:
        typer.echo("Dry run only; skipped upload.")
        return

    HfApi(token=token).upload_file(
        path_or_fileobj=str(input),
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"update features for {venue}/{year}",
    )
    typer.echo("Uploaded feature parquet.")


if __name__ == "__main__":
    app()
