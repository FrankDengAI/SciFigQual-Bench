#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars"]
# ///
"""Compute canonical corpus statistics — single source of truth for figures & docs."""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
from manifest_utils import load_consolidated  # noqa: E402

CONSOLIDATED = ROOT / "ccf-paper-figures-viewable" / "human_annotations_consolidated.jsonl"
PARQUET_ROOT = ROOT / "ccf-paper-figures" / "processed"
OUT = ROOT / "datasets" / "corpus_stats.json"


def _parquet_figure_keys() -> tuple[list[tuple[str, int, str]], Counter[str], set[str]]:
    keys: list[tuple[str, int, str]] = []
    venue_counts: Counter[str] = Counter()
    papers: set[str] = set()
    for path in sorted(PARQUET_ROOT.glob("*/202*/figures_clean.parquet")):
        df = pl.read_parquet(path, columns=["paper_id", "fig_index", "venue"])
        venue = str(df["venue"][0])
        for paper_id, fig_index in zip(df["paper_id"].to_list(), df["fig_index"].to_list()):
            keys.append((str(paper_id), int(fig_index), venue))
            venue_counts[venue] += 1
            papers.add(str(paper_id))
    return keys, venue_counts, papers


def compute() -> dict:
    records = load_consolidated(CONSOLIDATED)
    pq_keys, pq_venue_all, pq_papers = _parquet_figure_keys()
    pq_key_set = {(a, b) for a, b, _ in pq_keys}
    cons_key_set = {(str(r["paper_id"]), int(r["fig_index"])) for r in records}

    exportable = [r for r in records if (str(r["paper_id"]), int(r["fig_index"])) in pq_key_set]
    annotation_only = [r for r in records if (str(r["paper_id"]), int(r["fig_index"])) not in pq_key_set]
    unrated_keys = pq_key_set - cons_key_set

    unrated_by_venue: Counter[str] = Counter()
    for paper_id, fig_index, venue in pq_keys:
        if (paper_id, fig_index) in unrated_keys:
            unrated_by_venue[venue] += 1

    exportable_venue = Counter(str(r["venue"]) for r in exportable)
    annotated_venue = Counter(str(r["venue"]) for r in records)
    annotation_records = sum(len(r.get("annotations") or []) for r in records)

    clean_figures = len(pq_keys)
    annotated_figures = len(records)
    exportable_figures = len(exportable)
    unrated_figures = len(unrated_keys)
    annotation_only_figures = len(annotation_only)

    assert exportable_figures + unrated_figures == clean_figures, (
        f"{exportable_figures} + {unrated_figures} != {clean_figures}"
    )
    assert exportable_figures + annotation_only_figures == annotated_figures, (
        f"{exportable_figures} + {annotation_only_figures} != {annotated_figures}"
    )

    return {
        "built_at": datetime.now(UTC).isoformat(),
        "sources": {
            "consolidated": str(CONSOLIDATED),
            "parquet_glob": str(PARQUET_ROOT / "*/202*/figures_clean.parquet"),
        },
        "funnel": {
            "corpus_pdfs": 62694,
            "papers_with_figures": len(pq_papers),
            "clean_figures": clean_figures,
            "human_rated_exportable": exportable_figures,
            "unrated_figures": unrated_figures,
            "human_annotated_total": annotated_figures,
            "annotation_only_no_image": annotation_only_figures,
            "rated_venue_exportable": dict(exportable_venue),
            "rated_venue_annotated_all": dict(annotated_venue),
            "unrated_venue": dict(unrated_by_venue),
            "parquet_venue_all": dict(pq_venue_all),
        },
        "annotation": {
            "annotated_figures": annotated_figures,
            "exportable_figures": exportable_figures,
            "annotation_records": annotation_records,
            "annotated_papers": len({str(r["paper_id"]) for r in records}),
        },
        "vlm_eval": {
            "eval_subset_size": 1200,
            "eval_manifest": "configs/human_eval_subset_1200.jsonl",
            "eval_dataset_dir": "datasets/eval1200",
            "sampled_from": "human_rated_exportable (6308)",
        },
        "integrity_checks": {
            "exportable_plus_unrated_equals_clean": exportable_figures + unrated_figures == clean_figures,
            "exportable_plus_annotation_only_equals_annotated": exportable_figures + annotation_only_figures
            == annotated_figures,
            "rated_venue_exportable_sums": sum(exportable_venue.values()) == exportable_figures,
            "rated_venue_annotated_sums": sum(annotated_venue.values()) == annotated_figures,
        },
    }


def main() -> None:
    stats = compute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    f = stats["funnel"]
    print(f"Wrote {OUT}")
    print(
        f"Funnel: {f['corpus_pdfs']:,} PDFs -> {f['papers_with_figures']:,} papers -> "
        f"{f['clean_figures']:,} figures -> {f['human_rated_exportable']:,} rated + "
        f"{f['unrated_figures']:,} unrated (+{f['annotation_only_no_image']} annotation-only)"
    )
    print(f"Exportable venue: {f['rated_venue_exportable']}")


if __name__ == "__main__":
    main()
