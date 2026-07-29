#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars"]
# ///
"""Evaluate model scores against 16-annotator human means on the eval-800 subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from manifest_utils import (
    DEFAULT_CONSOLIDATED,
    DIMENSIONS,
    human_means_from_consolidated,
    join_model_human,
    load_manifest_df,
    summarize_alignment,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "configs" / "human_eval_subset_1200.jsonl"
DEFAULT_OUT = ROOT / "outputs" / "benchmark_eval1200" / "leaderboard"


def _load_results(path: Path) -> pl.DataFrame:
    if path.is_dir():
        frames = [pl.read_parquet(p) for p in sorted(path.rglob("results.parquet"))]
        if not frames:
            raise FileNotFoundError(f"No results.parquet under {path}")
        return pl.concat(frames, how="diagonal_relaxed")
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_parquet(path)


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def evaluate(
    *,
    results_path: Path,
    consolidated_path: Path,
    manifest_path: Path | None,
    output_dir: Path,
    run_label: str | None,
    api_calls_per_fig: int | None,
) -> pl.DataFrame:
    model_df = _load_results(results_path)
    human_all = human_means_from_consolidated(consolidated_path)
    if manifest_path is not None:
        manifest = load_manifest_df(manifest_path)
        human_df = human_all.join(manifest.select("paper_id", "fig_index"), on=["paper_id", "fig_index"], how="inner")
    else:
        human_df = human_all

    summaries: list[dict] = []
    if "source_name" in model_df.columns:
        if run_label:
            source_names = [run_label]
            model_df = model_df.filter(pl.col("source_name") == run_label)
        else:
            source_names = sorted(model_df["source_name"].drop_nulls().unique().to_list())
    else:
        source_names = [run_label or "model"]

    per_dim_rows: list[dict] = []
    for source in source_names:
        merged = join_model_human(model_df, human_df, source_name=None if source == "model" else source)
        label = source if source != "model" else (run_label or "model")
        summary = summarize_alignment(merged, run_label=label, api_calls_per_fig=api_calls_per_fig)
        summaries.append(summary)
        for dim in DIMENSIONS:
            metrics = {
                "run_label": label,
                "dimension": dim,
                "n": summary.get(f"{dim}_n"),
                "mae": summary.get(f"{dim}_mae"),
                "bias": summary.get(f"{dim}_bias"),
                "within1": summary.get(f"{dim}_within1"),
                "srcc": summary.get(f"{dim}_srcc"),
            }
            per_dim_rows.append(metrics)

    leaderboard = pl.DataFrame(summaries)
    per_dim = pl.DataFrame(per_dim_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.write_csv(output_dir / "leaderboard.csv")
    per_dim.write_csv(output_dir / "per_dimension.csv")

    md_rows = []
    for row in leaderboard.iter_rows(named=True):
        md_rows.append(
            [
                str(row["run_label"]),
                _fmt(row.get("overall_n"), 0),
                _fmt(row.get("overall_mae")),
                _fmt(row.get("overall_bias")),
                _fmt(row.get("overall_within1")),
                _fmt(row.get("overall_srcc")),
                _fmt(row.get("caption_consistency_mae")),
                _fmt(row.get("context_consistency_mae")),
                _fmt(row.get("misleading_risk_mae")),
            ]
        )
    markdown = "# Human Agreement Leaderboard\n\n"
    markdown += _markdown_table(
        ["Run", "n", "Overall MAE", "Bias", "Within-1", "SRCC", "CC MAE", "CTX MAE", "MR MAE"],
        md_rows,
    )
    markdown += "\n\n## Per-dimension\n\n"
    dim_rows = []
    for row in per_dim.iter_rows(named=True):
        dim_rows.append(
            [
                str(row["run_label"]),
                str(row["dimension"]),
                _fmt(row.get("n"), 0),
                _fmt(row.get("mae")),
                _fmt(row.get("bias")),
                _fmt(row.get("within1")),
                _fmt(row.get("srcc")),
            ]
        )
    markdown += _markdown_table(
        ["Run", "Dimension", "n", "MAE", "Bias", "Within-1", "SRCC"],
        dim_rows,
    )
    (output_dir / "leaderboard.md").write_text(markdown + "\n", encoding="utf-8")
    (output_dir / "leaderboard.json").write_text(
        json.dumps({"leaderboard": summaries, "per_dimension": per_dim_rows}, indent=2),
        encoding="utf-8",
    )
    return leaderboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model vs human means (eval-800)")
    parser.add_argument("--results", type=Path, required=True, help="results.parquet or run directory")
    parser.add_argument("--consolidated", type=Path, default=DEFAULT_CONSOLIDATED)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--run-label", type=str, default=None, help="Filter source_name / label for single run")
    parser.add_argument("--api-calls-per-fig", type=int, default=None)
    args = parser.parse_args()

    leaderboard = evaluate(
        results_path=args.results,
        consolidated_path=args.consolidated,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        run_label=args.run_label,
        api_calls_per_fig=args.api_calls_per_fig,
    )
    print(leaderboard)
    print(f"\nWrote leaderboard to {args.output_dir}")


if __name__ == "__main__":
    main()
