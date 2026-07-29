"""Shared helpers for human-eval subset manifests and scoring filters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

DIMENSIONS = [
    "visual_clarity",
    "structure_layout",
    "caption_consistency",
    "context_consistency",
    "misleading_risk",
]

DEFAULT_CONSOLIDATED = (
    Path(__file__).resolve().parents[2]
    / "ccf-paper-figures-viewable"
    / "human_annotations_consolidated.jsonl"
)


def load_consolidated(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_manifest_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def manifest_keys_from_records(records: list[dict[str, Any]]) -> set[tuple[str, int]]:
    return {(str(r["paper_id"]), int(r["fig_index"])) for r in records}


def load_manifest_keys(path: Path) -> set[tuple[str, int]]:
    return manifest_keys_from_records(load_consolidated(path))


def load_manifest_df(path: Path) -> pl.DataFrame:
    records = load_consolidated(path)
    if not records:
        return pl.DataFrame({"paper_id": [], "fig_index": []})
    rows = [
        {
            "paper_id": str(r["paper_id"]),
            "fig_index": int(r["fig_index"]),
            "venue": r.get("venue"),
            "year": int(r["year"]),
        }
        for r in records
    ]
    return pl.DataFrame(rows)


def filter_dataframe_by_manifest(df: pl.DataFrame, manifest_path: Path | None) -> pl.DataFrame:
    if manifest_path is None:
        return df
    manifest = load_manifest_df(manifest_path)
    if manifest.is_empty():
        return df.head(0)
    filtered = df.join(manifest.select("paper_id", "fig_index"), on=["paper_id", "fig_index"], how="inner")
    return filtered


def human_means_from_consolidated(path: Path) -> pl.DataFrame:
    """Aggregate per-figure human means from consolidated JSONL."""
    records = load_consolidated(path)
    rows: list[dict[str, Any]] = []
    for rec in records:
        anns = rec.get("annotations") or []
        if not anns:
            continue
        row: dict[str, Any] = {
            "paper_id": str(rec["paper_id"]),
            "fig_index": int(rec["fig_index"]),
            "venue": rec.get("venue"),
            "year": int(rec["year"]),
            "context_count": int(rec.get("context_count") or 0),
            "annotator_count": int(rec.get("annotator_count") or len(anns)),
        }
        for dim in DIMENSIONS + ["overall_score"]:
            vals = [a[dim] for a in anns if a.get(dim) is not None]
            row[f"human_{dim}"] = sum(vals) / len(vals) if vals else None
        rows.append(row)
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows)


def compute_alignment_metrics(
    merged: pl.DataFrame,
    *,
    model_col: str,
    human_col: str,
) -> dict[str, float | int | None]:
    work = merged.filter(
        pl.col(model_col).is_not_null() & pl.col(human_col).is_not_null()
    )
    n = work.height
    if n == 0:
        return {"n": 0, "mae": None, "bias": None, "within1": None, "srcc": None}
    diff = work[model_col] - work[human_col]
    mae = float(diff.abs().mean())
    bias = float(diff.mean())
    within1 = float((diff.abs() <= 1).mean())
    srcc = _spearman(work[model_col].to_list(), work[human_col].to_list())
    return {"n": n, "mae": mae, "bias": bias, "within1": within1, "srcc": srcc}


def _rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[index][1]:
            end += 1
        avg_rank = (index + end) / 2.0 + 1.0
        for pos in range(index, end + 1):
            ranks[indexed[pos][0]] = avg_rank
        index = end + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    if len(set(x)) <= 1 or len(set(y)) <= 1:
        return None
    try:
        from scipy.stats import spearmanr

        result = spearmanr(x, y).statistic
        if result is None or result != result:
            return None
        return float(result)
    except Exception:
        rx = _rank_values(x)
        ry = _rank_values(y)
        mx = sum(rx) / len(rx)
        my = sum(ry) / len(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den_x = sum((a - mx) ** 2 for a in rx) ** 0.5
        den_y = sum((b - my) ** 2 for b in ry) ** 0.5
        if den_x == 0 or den_y == 0:
            return None
        return float(num / (den_x * den_y))


def join_model_human(
    model_df: pl.DataFrame,
    human_df: pl.DataFrame,
    *,
    source_name: str | None = None,
) -> pl.DataFrame:
    work = model_df
    if source_name is not None and "source_name" in work.columns:
        work = work.filter(pl.col("source_name") == source_name)
    keys = ["paper_id", "fig_index"]
    human_cols = [f"human_{dim}" for dim in DIMENSIONS + ["overall_score"]]
    keep_human = [c for c in keys + human_cols if c in human_df.columns]
    return work.join(human_df.select(keep_human), on=keys, how="inner")


def summarize_alignment(
    merged: pl.DataFrame,
    *,
    run_label: str,
    api_calls_per_fig: int | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"run_label": run_label}
    if api_calls_per_fig is not None:
        row["api_calls_per_fig"] = api_calls_per_fig
        row["api_calls_total"] = api_calls_per_fig * merged.height if merged.height else 0

    for dim in ["overall_score"] + DIMENSIONS:
        model_col = "overall_score" if dim == "overall_score" else dim
        human_col = f"human_{dim}"
        metrics = compute_alignment_metrics(merged, model_col=model_col, human_col=human_col)
        prefix = "overall" if dim == "overall_score" else dim
        row[f"{prefix}_n"] = metrics["n"]
        row[f"{prefix}_mae"] = metrics["mae"]
        row[f"{prefix}_bias"] = metrics["bias"]
        row[f"{prefix}_within1"] = metrics["within1"]
        row[f"{prefix}_srcc"] = metrics["srcc"]
    return row
