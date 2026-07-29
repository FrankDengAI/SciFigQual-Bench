#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Deterministic stratified paper-level sample (exactly 800 figures).

Selection is fully reproducible without randomness:
  1. Sort papers by (venue, year, paper_id)
  2. Proportional venue×year allocation
  3. Deterministic swap adjustment to hit the exact figure target
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from manifest_utils import (
    DEFAULT_CONSOLIDATED,
    load_consolidated,
    write_manifest_jsonl,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "configs" / "human_eval_subset_800.jsonl"
DEFAULT_META = ROOT / "configs" / "human_eval_subset_800_meta.json"
SELECTION_VERSION = "deterministic-v1"


def _paper_groups(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        if int(rec.get("annotator_count") or 0) < 3:
            continue
        groups[str(rec["paper_id"])].append(rec)
    return groups


def _paper_meta(figs: list[dict]) -> dict:
    first = figs[0]
    ctx_figs = sum(1 for f in figs if int(f.get("context_count") or 0) > 0)
    return {
        "paper_id": str(first["paper_id"]),
        "venue": first["venue"],
        "year": int(first["year"]),
        "n_figs": len(figs),
        "ctx_figs": ctx_figs,
    }


def _allocate_targets(papers_by_cell: dict[tuple[str, int], list[dict]], total: int) -> dict[tuple[str, int], int]:
    cell_fig_counts = {cell: sum(p["n_figs"] for p in papers) for cell, papers in papers_by_cell.items()}
    grand = sum(cell_fig_counts.values())
    targets: dict[tuple[str, int], int] = {}
    allocated = 0
    cells = sorted(cell_fig_counts.keys())
    for index, cell in enumerate(cells):
        if index == len(cells) - 1:
            targets[cell] = max(0, total - allocated)
        else:
            share = round(total * cell_fig_counts[cell] / grand)
            targets[cell] = share
            allocated += share
    return targets


def _figure_count(chosen: set[str], groups: dict[str, list[dict]]) -> int:
    return sum(len(groups[paper_id]) for paper_id in chosen)


def _adjust_to_exact(
    chosen: set[str],
    all_papers: list[dict],
    groups: dict[str, list[dict]],
    target: int,
) -> set[str]:
    chosen = set(chosen)
    paper_by_id = {p["paper_id"]: p for p in all_papers}

    while _figure_count(chosen, groups) > target:
        current = _figure_count(chosen, groups)
        excess = current - target
        removed = False
        for rem_id in sorted(
            chosen,
            key=lambda pid: (paper_by_id[pid]["venue"], paper_by_id[pid]["year"], pid),
            reverse=True,
        ):
            if len(groups[rem_id]) == excess:
                chosen.remove(rem_id)
                removed = True
                break
        if removed:
            continue
        best_id = None
        best_distance = None
        for rem_id in sorted(
            chosen,
            key=lambda pid: (paper_by_id[pid]["venue"], paper_by_id[pid]["year"], pid),
            reverse=True,
        ):
            after = current - len(groups[rem_id])
            distance = abs(after - target)
            if after > target:
                continue
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_id = rem_id
        if best_id is not None:
            chosen.remove(best_id)
            continue
        rem_id = max(chosen, key=lambda pid: len(groups[pid]))
        chosen.remove(rem_id)

    remaining = [p for p in all_papers if p["paper_id"] not in chosen]
    remaining.sort(key=lambda p: (p["venue"], p["year"], p["paper_id"]))

    while _figure_count(chosen, groups) < target:
        current = _figure_count(chosen, groups)
        delta = target - current
        added = False
        for paper in remaining:
            if paper["paper_id"] in chosen:
                continue
            if paper["n_figs"] == delta:
                chosen.add(paper["paper_id"])
                added = True
                break
        if added:
            continue
        for paper in remaining:
            if paper["paper_id"] in chosen:
                continue
            if paper["n_figs"] < delta:
                chosen.add(paper["paper_id"])
                added = True
                break
        if added:
            continue
        for add in remaining:
            if add["paper_id"] in chosen:
                continue
            for rem_id in sorted(chosen, reverse=True):
                if add["n_figs"] - len(groups[rem_id]) == delta:
                    chosen.remove(rem_id)
                    chosen.add(add["paper_id"])
                    added = True
                    break
            if added:
                break
        if not added:
            break

    final = _figure_count(chosen, groups)
    if final != target:
        raise RuntimeError(f"Could not reach exactly {target} figures (got {final}).")
    return chosen


def sample_subset(records: list[dict], *, n_figures: int) -> list[dict]:
    groups = _paper_groups(records)
    all_papers = [_paper_meta(figs) for figs in groups.values()]
    papers_by_cell: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for paper in all_papers:
        papers_by_cell[(paper["venue"], paper["year"])].append(paper)
    for cell in papers_by_cell:
        papers_by_cell[cell].sort(key=lambda p: p["paper_id"])

    targets = _allocate_targets(papers_by_cell, n_figures)
    chosen: set[str] = set()
    for cell in sorted(papers_by_cell):
        target = targets[cell]
        count = 0
        for paper in papers_by_cell[cell]:
            if count >= target:
                break
            if count + paper["n_figs"] > target and count + paper["n_figs"] > target + 1:
                continue
            chosen.add(paper["paper_id"])
            count += paper["n_figs"]

    chosen = _adjust_to_exact(chosen, all_papers, groups, n_figures)
    subset: list[dict] = []
    for paper_id in sorted(chosen):
        subset.extend(groups[paper_id])
    subset.sort(key=lambda r: (r["venue"], r["year"], r["paper_id"], r["fig_index"]))
    if len(subset) != n_figures:
        raise RuntimeError(f"Expected {n_figures} figures, got {len(subset)}.")
    return subset


def build_meta(subset: list[dict], *, source: Path) -> dict:
    ann_dist = Counter(int(r.get("annotator_count") or 0) for r in subset)
    venue_dist = Counter(r["venue"] for r in subset)
    year_dist = Counter(int(r["year"]) for r in subset)
    ctx_with = sum(1 for r in subset if int(r.get("context_count") or 0) > 0)
    return {
        "built_at": datetime.now(UTC).isoformat(),
        "source": str(source),
        "selection_method": SELECTION_VERSION,
        "selection_rule": "sorted (venue, year, paper_id); paper-level all-or-nothing; no randomness",
        "figure_count": len(subset),
        "annotation_count": sum(int(r.get("annotator_count") or 0) for r in subset),
        "paper_count": len({r["paper_id"] for r in subset}),
        "annotator_count_distribution": dict(sorted(ann_dist.items())),
        "venue_distribution": dict(venue_dist),
        "year_distribution": {str(k): v for k, v in sorted(year_dist.items())},
        "with_context_figures": ctx_with,
        "without_context_figures": len(subset) - ctx_with,
        "with_context_ratio": round(ctx_with / len(subset), 4) if subset else 0.0,
        "human_gold_standard": {
            "join_key": ["paper_id", "fig_index"],
            "aggregation": "mean over annotations[] for each dimension and overall_score",
            "annotator_filter": "none (all annotators in consolidated pool)",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic human-eval subset manifest")
    parser.add_argument("--input", type=Path, default=DEFAULT_CONSOLIDATED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--n-figures", type=int, default=800)
    args = parser.parse_args()

    records = load_consolidated(args.input)
    subset = sample_subset(records, n_figures=args.n_figures)
    write_manifest_jsonl(subset, args.output)
    meta = build_meta(subset, source=args.input)
    args.meta.parent.mkdir(parents=True, exist_ok=True)
    args.meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(subset)} figures to {args.output}")
    print(f"Meta: {args.meta}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
