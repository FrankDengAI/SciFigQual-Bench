"""End-to-end feature extraction from cleaned figure shards."""

from __future__ import annotations

from typing import Any

import polars as pl
from PIL import UnidentifiedImageError

from .base import (
    compute_blur_laplacian_var,
    compute_pixel_count_mp,
    compute_whitespace_ratio,
    decode_image_bytes,
)
from .ocr import BaseOCRBackend, OCRBackendName, get_ocr_backend
from .schema import FEATURE_SCHEMA, FEATURE_VERSION


def _base_record(row: dict[str, Any], ocr_backend_name: str) -> dict[str, Any]:
    caption = (row.get("caption") or "").strip()
    return {
        "paper_id": row["paper_id"],
        "venue": row["venue"],
        "year": row["year"],
        "fig_index": row["fig_index"],
        "feature_version": FEATURE_VERSION,
        "ocr_backend": ocr_backend_name,
        "decode_ok": False,
        "decode_error": None,
        "has_caption": bool(caption),
        "caption_length_chars": len(caption),
        "pixel_count_mp": None,
        "blur_laplacian_var": None,
        "whitespace_ratio": None,
        "has_readable_text": None,
        "ocr_mean_confidence": None,
        "text_box_count": None,
        "text_region_ratio": None,
        "panel_label_count": None,
        "caption_ocr_token_overlap_ratio": None,
        "caption_numeric_overlap_ratio": None,
    }


def _extract_row_features(row: dict[str, Any], ocr_backend: BaseOCRBackend) -> dict[str, Any]:
    record = _base_record(row, ocr_backend.name)
    caption = (row.get("caption") or "").strip()
    image_bytes = row["image_bytes"]

    try:
        image_rgb = decode_image_bytes(image_bytes)
    except (UnidentifiedImageError, OSError, ValueError, TypeError) as exc:
        record["decode_error"] = f"{type(exc).__name__}: {exc}"
        return record

    ocr_features = ocr_backend.extract(image_rgb=image_rgb, caption=caption)

    record.update(
        {
            "decode_ok": True,
            "pixel_count_mp": compute_pixel_count_mp(image_rgb),
            "blur_laplacian_var": compute_blur_laplacian_var(image_rgb),
            "whitespace_ratio": compute_whitespace_ratio(image_rgb),
            "has_readable_text": ocr_features.has_readable_text,
            "ocr_mean_confidence": ocr_features.ocr_mean_confidence,
            "text_box_count": ocr_features.text_box_count,
            "text_region_ratio": ocr_features.text_region_ratio,
            "panel_label_count": ocr_features.panel_label_count,
            "caption_ocr_token_overlap_ratio": ocr_features.caption_ocr_token_overlap_ratio,
            "caption_numeric_overlap_ratio": ocr_features.caption_numeric_overlap_ratio,
        }
    )
    return record


def build_feature_dataframe(
    figures_df: pl.DataFrame,
    *,
    ocr_backend_name: OCRBackendName = "paddleocr",
) -> pl.DataFrame:
    required_columns = {"paper_id", "venue", "year", "fig_index", "caption", "image_bytes"}
    missing = sorted(required_columns - set(figures_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ocr_backend = get_ocr_backend(ocr_backend_name)
    rows = [
        _extract_row_features(row, ocr_backend=ocr_backend)
        for row in figures_df.iter_rows(named=True)
    ]
    if not rows:
        return pl.DataFrame(schema=FEATURE_SCHEMA)
    return pl.DataFrame(rows, schema=FEATURE_SCHEMA)
