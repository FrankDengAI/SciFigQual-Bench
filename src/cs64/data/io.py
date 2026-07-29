"""Shared local/Hugging Face data loading helpers."""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional convenience dependency
    load_dotenv = None

DEFAULT_HF_REPO_ID = "ccf-team/ccf-paper-figures"


def load_project_env(project_root: Path | None = None) -> None:
    """Load environment variables from the project-level .env file if present."""
    root = project_root or Path(__file__).resolve().parents[3]
    env_path = root / ".env"
    if not env_path.exists():
        return

    if load_dotenv is not None:
        load_dotenv(env_path)
        return

    # Fallback parser so project commands can still use `.env` without
    # requiring python-dotenv in the runtime environment.
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def build_hf_figures_uri(repo_id: str, venue: str, year: int, stage: str = "clean") -> str:
    if stage not in {"clean", "raw"}:
        raise ValueError("stage must be 'clean' or 'raw'")
    fname = "figures_clean.parquet" if stage == "clean" else "figures_raw.parquet"
    return f"hf://datasets/{repo_id}/processed/{venue}/{year}/{fname}"


def get_env_token(env_var: str = "HF_TOKEN") -> str:
    token = os.environ.get(env_var, "").strip()
    if not token:
        raise ValueError(f"Missing required token in environment variable: {env_var}")
    return token


def resolve_figure_input_source(
    *,
    input_path: Path | None = None,
    repo_id: str | None = None,
    venue: str | None = None,
    year: int | None = None,
    stage: str = "clean",
) -> tuple[str, dict[str, str] | None]:
    """Resolve either a local parquet path or an HF dataset shard URI."""
    has_local = input_path is not None
    has_hf = any(value is not None for value in (repo_id, venue, year))

    if has_local and has_hf:
        raise ValueError("Provide either --input or --repo-id/--venue/--year, not both")
    if not has_local and not has_hf:
        raise ValueError("Provide either --input or --repo-id/--venue/--year")

    if has_local:
        return str(input_path), None

    if repo_id is None or venue is None or year is None:
        raise ValueError("Hugging Face input requires --repo-id, --venue, and --year")

    return build_hf_figures_uri(repo_id=repo_id, venue=venue, year=year, stage=stage), {
        "token": get_env_token("HF_TOKEN")
    }


def load_figure_dataframe(
    *,
    input_path: Path | None = None,
    repo_id: str | None = None,
    venue: str | None = None,
    year: int | None = None,
    stage: str = "clean",
) -> pl.DataFrame:
    uri, storage_options = resolve_figure_input_source(
        input_path=input_path,
        repo_id=repo_id,
        venue=venue,
        year=year,
        stage=stage,
    )
    return pl.read_parquet(uri, storage_options=storage_options)
