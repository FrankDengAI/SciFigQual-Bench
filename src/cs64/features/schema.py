"""Schema definitions for figure feature extraction outputs."""

from __future__ import annotations

import polars as pl

FEATURE_VERSION = "v0.1.0"

FEATURE_SCHEMA: dict[str, pl.DataType] = {
    "paper_id": pl.String,
    "venue": pl.String,
    "year": pl.Int64,
    "fig_index": pl.Int64,
    "feature_version": pl.String,
    "ocr_backend": pl.String,
    "decode_ok": pl.Boolean,
    "decode_error": pl.String,
    "has_caption": pl.Boolean,
    "caption_length_chars": pl.Int32,
    "pixel_count_mp": pl.Float64,
    "blur_laplacian_var": pl.Float64,
    "whitespace_ratio": pl.Float64,
    "has_readable_text": pl.Boolean,
    "ocr_mean_confidence": pl.Float64,
    "text_box_count": pl.Int32,
    "text_region_ratio": pl.Float64,
    "panel_label_count": pl.Int32,
    "caption_ocr_token_overlap_ratio": pl.Float64,
    "caption_numeric_overlap_ratio": pl.Float64,
}
