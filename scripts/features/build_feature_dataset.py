#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "opencv-python",
#   "pillow",
#   "polars",
#   "pyarrow",
#   "python-dotenv",
#   "typer",
# ]
# ///
"""Build a figure-level feature dataset from figures_clean.parquet."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cs64.data import DEFAULT_HF_REPO_ID, load_figure_dataframe, load_project_env
from cs64.features import build_feature_dataframe

app = typer.Typer(add_completion=False, pretty_exceptions_short=True)


def _limit_to_first_papers(df: pl.DataFrame, paper_limit: int | None) -> pl.DataFrame:
    if paper_limit is None:
        return df
    if paper_limit <= 0:
        return df.head(0)
    first_paper_ids = (
        df.select("paper_id")
        .unique()
        .sort("paper_id")
        .head(paper_limit)
        .get_column("paper_id")
        .to_list()
    )
    return df.filter(pl.col("paper_id").is_in(first_paper_ids))


@app.command()
def main(
    output: Annotated[Path, typer.Option(help="Path to figures_features.parquet")],
    input: Annotated[Path | None, typer.Option(help="Path to local figures parquet")] = None,
    ocr_backend: Annotated[str, typer.Option(help="OCR backend name")] = "paddleocr",
    repo_id: Annotated[str, typer.Option(help="HF dataset repo id")] = DEFAULT_HF_REPO_ID,
    venue: Annotated[str | None, typer.Option(help="HF shard venue")] = None,
    year: Annotated[int | None, typer.Option(help="HF shard year")] = None,
    stage: Annotated[str, typer.Option(help="HF shard stage: clean or raw")] = "clean",
    limit: Annotated[int | None, typer.Option(help="Optional max number of rows")] = None,
    paper_limit: Annotated[
        int | None,
        typer.Option(help="Optional limit on the first N papers by paper_id sort order"),
    ] = None,
    include_excluded: Annotated[
        bool,
        typer.Option(
            help=(
                "Keep rows flagged exclude_from_scoring=true. Defaults to true so "
                "feature coverage matches model scoring and the annotation frontend."
            )
        ),
    ] = True,
) -> None:
    load_project_env()
    df = load_figure_dataframe(
        input_path=input,
        repo_id=None if input is not None else repo_id,
        venue=venue,
        year=year,
        stage=stage,
    )

    if not include_excluded and "exclude_from_scoring" in df.columns:
        df = df.filter(~pl.col("exclude_from_scoring").fill_null(False))
    df = _limit_to_first_papers(df, paper_limit)
    if limit is not None:
        df = df.head(limit)

    features_df = build_feature_dataframe(df, ocr_backend_name=ocr_backend)
    output.parent.mkdir(parents=True, exist_ok=True)
    features_df.write_parquet(output)

    typer.echo(f"Wrote {len(features_df)} feature rows -> {output}")


if __name__ == "__main__":
    app()
