from __future__ import annotations

import json
import re
from typing import Any

import polars as pl


def normalize_paper_id(value: Any) -> str:
    return re.sub(r"^\d+_", "", str(value or ""))


def parse_fig_num(value: Any) -> int | None:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else None


def truncate_text(text: Any, max_chars: int = 500) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max(0, max_chars - 3)].rstrip() + "..."


def load_context_table(path: str | None) -> pl.DataFrame:
    if not path:
        return pl.DataFrame()
    return pl.read_parquet(path)


def context_lookup_map(context: pl.DataFrame) -> dict[tuple[str, int], list[dict[str, Any]]]:
    lookup: dict[tuple[str, int], list[dict[str, Any]]] = {}
    if context.is_empty() or not {"paper_id", "fig_num"}.issubset(context.columns):
        return lookup
    work = context.with_columns(
        pl.col("paper_id")
        .map_elements(normalize_paper_id, return_dtype=pl.String)
        .alias("paper_id_norm"),
        pl.col("fig_num")
        .map_elements(parse_fig_num, return_dtype=pl.Int64)
        .alias("fig_index_from_fig_num"),
    )
    for row in work.drop_nulls("fig_index_from_fig_num").iter_rows(named=True):
        key = (str(row["paper_id_norm"]), int(row["fig_index_from_fig_num"]))
        lookup.setdefault(key, []).append(
            {
                "fig_num": row.get("fig_num"),
                "section": row.get("section") or "Unknown section",
                "pages": row.get("context_pages") or [],
                "texts": row.get("context_texts") or [],
            }
        )
    return lookup


def build_context_payload(contexts: list[dict[str, Any]], max_snippets: int = 2) -> dict[str, Any]:
    snippets: list[dict[str, Any]] = []
    for item in contexts:
        for text in item.get("texts") or []:
            clean = truncate_text(text)
            if clean:
                snippets.append(
                    {
                        "fig_num": item.get("fig_num"),
                        "section": item.get("section"),
                        "pages": [str(page) for page in item.get("pages") or []],
                        "text": clean,
                    }
                )
    return {
        "matched_reference_rows": len(contexts),
        "usable_text_snippets": len(snippets),
        "selected_contexts": snippets[:max_snippets],
    }


def context_key(row: dict[str, Any]) -> tuple[str, int]:
    return (normalize_paper_id(row.get("paper_id")), int(row["fig_index"]))


def has_caption(row: dict[str, Any]) -> bool:
    return bool(str(row.get("caption") or "").strip())


def has_any_text_evidence(row: dict[str, Any], contexts: list[dict[str, Any]]) -> bool:
    if has_caption(row):
        return True
    return build_context_payload(contexts)["usable_text_snippets"] > 0


def filter_rows_with_text_evidence(
    df: pl.DataFrame,
    context_lookup: dict[tuple[str, int], list[dict[str, Any]]],
) -> tuple[pl.DataFrame, int]:
    if df.is_empty():
        return df, 0
    rows = [
        row
        for row in df.iter_rows(named=True)
        if has_any_text_evidence(row, context_lookup.get(context_key(row), []))
    ]
    skipped = len(df) - len(rows)
    if not rows:
        return pl.DataFrame(schema=df.schema), skipped
    return pl.DataFrame(rows, schema=df.schema), skipped


def optional_row_field(row: dict[str, Any], names: list[str]) -> dict[str, Any]:
    return {name: row.get(name) for name in names if name in row and row.get(name) is not None}


def figure_map_entry(row: dict[str, Any], contexts: list[dict[str, Any]]) -> dict[str, Any]:
    context_payload = build_context_payload(contexts, max_snippets=1)
    sections = sorted(
        {
            str(snippet.get("section"))
            for snippet in context_payload["selected_contexts"]
            if snippet.get("section")
        }
    )
    return {
        "fig_index": int(row["fig_index"]),
        "caption_short": truncate_text(row.get("caption"), max_chars=220),
        "location": optional_row_field(
            row,
            ["page_num", "page", "cross_page", "content_type", "caption_matched"],
        ),
        "context_sections": sections,
    }


def target_figure_payload(
    row: dict[str, Any],
    *,
    feature_payload: dict[str, Any] | None,
    contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "fig_index": int(row["fig_index"]),
        "caption": row.get("caption") or "",
        **optional_row_field(row, ["figure_type"]),
        "location": optional_row_field(
            row,
            ["page_num", "page", "cross_page", "content_type", "caption_matched"],
        ),
        "context": build_context_payload(contexts),
        "feature_json": feature_payload,
    }


def paper_prompt_payload(
    *,
    paper_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    feature_lookup: dict[tuple[str, int], dict[str, Any]],
    context_lookup: dict[tuple[str, int], list[dict[str, Any]]],
    prior_assessments: list[dict[str, Any]],
    include_paper_batch_context: bool = True,
) -> dict[str, Any]:
    first = paper_rows[0]
    paper_id = str(first["paper_id"])
    figure_map = []
    target_figures = []
    for row in paper_rows:
        key = (normalize_paper_id(row.get("paper_id")), int(row["fig_index"]))
        contexts = context_lookup.get(key, [])
        figure_map.append(figure_map_entry(row, contexts))
    for row in target_rows:
        feature_key = (str(row["paper_id"]), int(row["fig_index"]))
        context_key = (normalize_paper_id(row.get("paper_id")), int(row["fig_index"]))
        target_figures.append(
            target_figure_payload(
                row,
                feature_payload=feature_lookup.get(feature_key),
                contexts=context_lookup.get(context_key, []),
            )
        )
    payload = {
        "paper": {
            "paper_id": paper_id,
            "title": first.get("title") or "",
            "venue": first.get("venue"),
            "year": first.get("year"),
        },
        "target_figures": sorted(target_figures, key=lambda item: item["fig_index"]),
    }
    if include_paper_batch_context:
        payload["figure_map"] = sorted(figure_map, key=lambda item: item["fig_index"])
        payload["prior_figure_assessments"] = prior_assessments
    return payload


def format_paper_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
