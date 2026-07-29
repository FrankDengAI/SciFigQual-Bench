from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import typer


def _normalize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in value.items()}
    return value


def load_feature_table(path: str | Path) -> pl.DataFrame:
    table = pl.read_parquet(path)
    missing = sorted({"paper_id", "fig_index"} - set(table.columns))
    if missing:
        raise typer.BadParameter(f"Feature parquet missing columns: {missing}")
    return table


def feature_lookup_map(table: pl.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in table.iter_rows(named=True):
        payload = {
            key: _normalize(value)
            for key, value in row.items()
            if key not in {"paper_id", "fig_index", "venue", "year"}
        }
        lookup[(str(row["paper_id"]), int(row["fig_index"]))] = payload
    return lookup


def format_feature_json(payload: dict[str, Any] | None) -> str:
    if payload is None:
        return "null"
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
