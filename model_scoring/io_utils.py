from __future__ import annotations

import io
import json
import re
from pathlib import Path

import polars as pl
import typer
from dotenv import load_dotenv
from huggingface_hub import HfApi

REPO_ID = "ccf-team/ccf-paper-figures"
HF_BASE = f"hf://datasets/{REPO_ID}"
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "model_scoring"
EXPLANATION_COLUMNS = [
    "summary",
    "visual_clarity_reason",
    "structure_layout_reason",
    "caption_consistency_reason",
    "context_consistency_reason",
    "misleading_risk_reason",
]
SCORE_COLUMNS = {
    "visual_clarity": pl.Float32,
    "structure_layout": pl.Float32,
    "caption_consistency": pl.Float32,
    "context_consistency": pl.Float32,
    "misleading_risk": pl.Float32,
    "overall_score": pl.Float32,
}
SOURCE_COLUMNS = {
    "source_type": pl.String,
    "source_name": pl.String,
}

RESULT_COLUMN_DTYPES = {
    **SOURCE_COLUMNS,
    **SCORE_COLUMNS,
    **{column: pl.String for column in EXPLANATION_COLUMNS},
}


def load_env() -> None:
    load_dotenv(ENV_PATH)


def safe_filename(label: str) -> str:
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "-", label).strip().strip(".")
    return cleaned or "model-output"


def load_figures_from_hf(venue: str, year: int, stage: str, token: str) -> pl.DataFrame:
    fname = "figures_clean.parquet" if stage == "clean" else "figures_raw.parquet"
    return pl.read_parquet(
        f"{HF_BASE}/processed/{venue}/{year}/{fname}",
        storage_options={"token": token},
    )


def load_features_from_hf(venue: str, year: int, token: str) -> pl.DataFrame:
    return pl.read_parquet(
        f"{HF_BASE}/processed/{venue}/{year}/features/figures_features.parquet",
        storage_options={"token": token},
    )


def load_context_from_hf(venue: str, year: int, token: str) -> pl.DataFrame:
    return pl.read_parquet(
        f"{HF_BASE}/processed/{venue}/{year}/figures_context.parquet",
        storage_options={"token": token},
    )


def load_figures(
    parquet: str | None, venue: str, year: int, stage: str, token: str | None
) -> pl.DataFrame:
    if parquet:
        return pl.read_parquet(parquet)
    if not token:
        raise typer.BadParameter("HF_TOKEN is required when reading from HF.")
    return load_figures_from_hf(venue, year, stage, token)


def load_existing_results(venue: str, year: int, token: str | None) -> pl.DataFrame:
    if not token:
        return pl.DataFrame()
    try:
        df = pl.read_parquet(
            f"{HF_BASE}/processed/{venue}/{year}/results.parquet",
            storage_options={"token": token},
        )
        return normalize_result_columns(df)
    except Exception:
        return pl.DataFrame()


def normalize_result_columns(df: pl.DataFrame) -> pl.DataFrame:
    work = df
    for column, dtype in RESULT_COLUMN_DTYPES.items():
        if column not in work.columns:
            work = work.with_columns(pl.lit(None, dtype=dtype).alias(column))
        else:
            work = work.with_columns(pl.col(column).cast(dtype))
    return work


def merge_results(existing: pl.DataFrame, fresh: pl.DataFrame) -> pl.DataFrame:
    if existing.is_empty():
        return normalize_result_columns(fresh)
    existing = normalize_result_columns(existing)
    fresh = normalize_result_columns(fresh)
    key_expr = (
        pl.col("paper_id")
        + pl.lit("::")
        + pl.col("fig_index").cast(pl.String)
        + pl.lit("::")
        + pl.col("source_type")
        + pl.lit("::")
        + pl.col("source_name")
    )
    fresh_keys = fresh.with_columns(_merge_key=key_expr).get_column("_merge_key").to_list()
    kept_existing = existing.filter(~key_expr.is_in(fresh_keys).fill_null(False))
    return pl.concat([kept_existing, fresh], how="diagonal")


def upload_results(df: pl.DataFrame, venue: str, year: int, token: str) -> None:
    buf = io.BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    HfApi(token=token).upload_file(
        path_or_fileobj=buf,
        path_in_repo=f"processed/{venue}/{year}/results.parquet",
        repo_id=REPO_ID,
        repo_type="dataset",
    )


def audit_path(provider: str, venue: str, year: int, model_label: str) -> Path:
    return OUTPUT_ROOT / provider / venue / str(year) / f"{safe_filename(model_label)}.jsonl"


def append_result_jsonl(path: str | Path, record: dict) -> None:
    """Append one scored row to a JSONL file (incremental checkpoint)."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_audit(
    records: list[dict], provider: str, venue: str, year: int, model_label: str
) -> Path:
    path = audit_path(provider, venue, year, model_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def error_audit_path(provider: str, venue: str, year: int, model_label: str) -> Path:
    return OUTPUT_ROOT / provider / venue / str(year) / f"{safe_filename(model_label)}.errors.jsonl"


def write_error_audit(
    records: list[dict], provider: str, venue: str, year: int, model_label: str
) -> Path | None:
    if not records:
        return None
    path = error_audit_path(provider, venue, year, model_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def prepare_rows(df: pl.DataFrame, limit: int | None, paper_id: str | None) -> pl.DataFrame:
    required = ["paper_id", "venue", "year", "fig_index", "caption", "image_bytes"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise typer.BadParameter(f"Input parquet missing columns: {missing}")
    work = df
    if paper_id:
        work = work.filter(pl.col("paper_id") == paper_id)
    if limit is not None and limit >= 0:
        work = work.head(limit)
    before = len(work)
    work = work.unique(subset=["paper_id", "fig_index"], keep="first", maintain_order=True)
    skipped = before - len(work)
    if skipped:
        typer.echo(f"Skipped {skipped} duplicate (paper_id, fig_index) rows.")
    return work


def limit_to_first_papers(df: pl.DataFrame, paper_limit: int | None) -> pl.DataFrame:
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


def filter_by_manifest(df: pl.DataFrame, manifest_path: str | Path | None) -> pl.DataFrame:
    if not manifest_path:
        return df
    path = Path(manifest_path)
    if not path.exists():
        raise typer.BadParameter(f"Manifest not found: {path}")
    manifest = pl.read_ndjson(path) if path.suffix == ".jsonl" else pl.read_json(path)
    required = {"paper_id", "fig_index"}
    missing = required - set(manifest.columns)
    if missing:
        raise typer.BadParameter(f"Manifest missing columns: {sorted(missing)}")
    manifest = manifest.select(
        pl.col("paper_id").cast(pl.String),
        pl.col("fig_index").cast(pl.Int32),
    ).unique()
    filtered = df.join(manifest, on=["paper_id", "fig_index"], how="inner")
    typer.echo(f"Manifest filter: {filtered.height}/{df.height} rows kept from {path.name}")
    return filtered
