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

import io
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO_ID = "ccf-team/ccf-paper-figures"
HF_BASE = f"hf://datasets/{REPO_ID}"
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
BACKUP_ROOT = REPO_ROOT / "outputs" / "model_scoring" / "deletions"
VALID_SOURCE_TYPES = {"model", "human"}

app = typer.Typer(add_completion=False, pretty_exceptions_short=True)


def _load_env() -> None:
    load_dotenv(ENV_PATH)


def _token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise typer.BadParameter("HF_TOKEN is required.")
    return token


def _load_results(venue: str, year: int, token: str) -> pl.DataFrame:
    return pl.read_parquet(
        f"{HF_BASE}/processed/{venue}/{year}/results.parquet",
        storage_options={"token": token},
    )


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    safe = re.sub(r"-+", "-", safe).strip(".-")
    return safe or "source"


def _backup_path(venue: str, year: int, source_type: str, source_name: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe = _safe_name(source_name)
    return BACKUP_ROOT / venue / str(year) / f"{source_type}-{safe}.{stamp}.parquet"


def _upload_results(df: pl.DataFrame, venue: str, year: int, token: str) -> None:
    buf = io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    HfApi(token=token).upload_file(
        path_or_fileobj=buf,
        path_in_repo=f"processed/{venue}/{year}/results.parquet",
        repo_id=REPO_ID,
        repo_type="dataset",
    )


def _resolve_source(
    source_type: str | None,
    source_name: str | None,
    model_label: str | None,
) -> tuple[str, str]:
    if model_label:
        if source_type and source_type != "model":
            raise typer.BadParameter("--model-label can only be used with source_type=model.")
        if source_name and source_name != model_label:
            raise typer.BadParameter("--source-name conflicts with --model-label.")
        source_type = "model"
        source_name = model_label

    if not source_type:
        raise typer.BadParameter("--source-type is required. Use model or human.")
    if source_type not in VALID_SOURCE_TYPES:
        allowed = ", ".join(sorted(VALID_SOURCE_TYPES))
        raise typer.BadParameter(f"--source-type must be one of: {allowed}.")
    if not source_name:
        raise typer.BadParameter("--source-name is required.")
    return source_type, source_name


@app.command()
def main(
    venue: Annotated[str, typer.Option(help="Conference name, e.g. ACL")],
    year: Annotated[int, typer.Option(help="Conference year")],
    source_type: Annotated[
        str | None,
        typer.Option("--source-type", help="Result source type: model or human"),
    ] = None,
    source_name: Annotated[
        str | None,
        typer.Option("--source-name", help="Exact source_name to delete"),
    ] = None,
    model_label: Annotated[
        str | None,
        typer.Option(
            "--model-label",
            help="Legacy alias for --source-type model --source-name",
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Preview rows that would be deleted without uploading")
    ] = False,
) -> None:
    source_type, source_name = _resolve_source(source_type, source_name, model_label)
    _load_env()
    token = _token()

    results = _load_results(venue, year, token)
    required = {"source_type", "source_name"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise typer.BadParameter(
            f"results.parquet is missing required columns: {', '.join(missing)}"
        )

    target = (pl.col("source_type") == source_type) & (pl.col("source_name") == source_name)
    to_delete = results.filter(target)
    if to_delete.is_empty():
        typer.echo(f"No rows found for {source_type}:{source_name}")
        raise typer.Exit()

    kept = results.filter(~target)
    typer.echo(f"Found {len(to_delete)} rows for {source_type}:{source_name}")
    typer.echo(f"Rows after deletion: {len(kept)}")

    backup_path = _backup_path(venue, year, source_type, source_name)
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    to_delete.write_parquet(backup_path)
    typer.echo(f"Backup written to {backup_path}")

    if dry_run:
        typer.echo("Dry run only. Skipped HF upload.")
        raise typer.Exit()

    _upload_results(kept, venue, year, token)
    typer.echo(f"Uploaded updated processed/{venue}/{year}/results.parquet to {REPO_ID}")


if __name__ == "__main__":
    app()
