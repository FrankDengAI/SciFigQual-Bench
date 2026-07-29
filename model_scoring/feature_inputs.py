from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import typer

FEATURE_FIELDS = [
    "feature_version",
    "ocr_backend",
    "decode_ok",
    "decode_error",
    "has_caption",
    "caption_length_chars",
    "pixel_count_mp",
    "blur_laplacian_var",
    "whitespace_ratio",
    "has_readable_text",
    "ocr_mean_confidence",
    "text_box_count",
    "text_region_ratio",
    "panel_label_count",
    "caption_ocr_token_overlap_ratio",
    "caption_numeric_overlap_ratio",
]

BASE_SIGNAL_FIELDS = [
    "feature_version",
    "ocr_backend",
    "decode_ok",
    "decode_error",
    "has_caption",
    "caption_length_chars",
    "has_readable_text",
]

DIMENSION_SIGNAL_FIELDS = {
    "visual_clarity": [
        "pixel_count_mp",
        "blur_laplacian_var",
        "ocr_mean_confidence",
        "text_region_ratio",
    ],
    "structure_layout": [
        "whitespace_ratio",
        "text_box_count",
        "panel_label_count",
    ],
    "caption_consistency": [
        "caption_ocr_token_overlap_ratio",
        "caption_numeric_overlap_ratio",
    ],
    "misleading_risk": [],
}


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def load_feature_table(path: str | Path) -> pl.DataFrame:
    table = pl.read_parquet(path)
    required = {"paper_id", "fig_index"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise typer.BadParameter(f"Feature parquet missing columns: {missing}")
    return table


def _feature_payload(row: dict[str, Any]) -> dict[str, Any]:
    base_signals = {field: _normalize_scalar(row.get(field)) for field in BASE_SIGNAL_FIELDS}
    dimension_signals = {
        dim: {field: _normalize_scalar(row.get(field)) for field in fields}
        for dim, fields in DIMENSION_SIGNAL_FIELDS.items()
    }
    return {
        "base_signals": base_signals,
        "dimension_signals": dimension_signals,
    }


def feature_lookup_map(table: pl.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
    rows = table.iter_rows(named=True)
    lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["paper_id"]), int(row["fig_index"]))
        lookup[key] = _feature_payload(row)
    return lookup


def format_feature_json(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
