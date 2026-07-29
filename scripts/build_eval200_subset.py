#!/usr/bin/env python3
"""Build datasets/eval200 from existing eval1200 (stratified subsample).

Used when human_annotations_consolidated.jsonl is unavailable.
Sampling: proportional venue×year allocation, deterministic paper order.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL1200_DIR = ROOT / "datasets" / "eval1200"
EVAL200_DIR = ROOT / "datasets" / "eval200"
CONFIGS = ROOT / "configs"
MANIFEST1200 = CONFIGS / "human_eval_subset_1200.jsonl"
FIGURES1200 = EVAL1200_DIR / "figures.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _allocate_targets(records: list[dict], total: int) -> dict[tuple[str, int], int]:
    cell_counts: dict[tuple[str, int], int] = Counter(
        (str(r["venue"]), int(r["year"])) for r in records
    )
    grand = sum(cell_counts.values())
    cells = sorted(cell_counts.keys())
    targets: dict[tuple[str, int], int] = {}
    allocated = 0
    for index, cell in enumerate(cells):
        if index == len(cells) - 1:
            targets[cell] = max(0, total - allocated)
        else:
            share = round(total * cell_counts[cell] / grand)
            targets[cell] = share
            allocated += share
    return targets


def sample_eval200(records: list[dict], n: int = 200) -> list[dict]:
    records = sorted(records, key=lambda r: (r["venue"], r["year"], r["paper_id"], r["fig_index"]))
    targets = _allocate_targets(records, n)
    by_cell: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for rec in records:
        by_cell[(str(rec["venue"]), int(rec["year"]))].append(rec)

    chosen: list[dict] = []
    for cell in sorted(by_cell.keys()):
        target = targets.get(cell, 0)
        chosen.extend(by_cell[cell][:target])

    if len(chosen) < n:
        chosen_keys = {(r["paper_id"], r["fig_index"]) for r in chosen}
        for rec in records:
            key = (rec["paper_id"], rec["fig_index"])
            if key not in chosen_keys:
                chosen.append(rec)
                chosen_keys.add(key)
            if len(chosen) >= n:
                break
    elif len(chosen) > n:
        chosen = chosen[:n]

    return sorted(chosen, key=lambda r: (r["venue"], r["year"], r["paper_id"], r["fig_index"]))


def _write_human_means_csv(records: list[dict], path: Path) -> None:
    fields = [
        "figure_id", "paper_id", "fig_index", "venue", "year", "image", "caption",
        "context_count", "annotator_count",
        "human_visual_clarity", "human_structure_layout", "human_caption_consistency",
        "human_context_consistency", "human_misleading_risk", "human_overall_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build eval200 from eval1200")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--source-figures", type=Path, default=FIGURES1200)
    parser.add_argument("--source-manifest", type=Path, default=MANIFEST1200)
    parser.add_argument("--out-dir", type=Path, default=EVAL200_DIR)
    args = parser.parse_args()

    if not args.source_figures.exists():
        raise FileNotFoundError(f"Missing {args.source_figures}")

    figures1200 = _load_jsonl(args.source_figures)
    selected = sample_eval200(figures1200, n=args.n)
    selected_keys = {(str(r["paper_id"]), int(r["fig_index"])) for r in selected}

    # Copy images
    images_out = args.out_dir / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    missing_images = 0
    for rec in selected:
        src_name = Path(str(rec["image"])).name
        src = EVAL1200_DIR / "images" / src_name
        dst = images_out / src_name
        if src.exists():
            if not dst.exists():
                shutil.copy2(src, dst)
        else:
            missing_images += 1

    # Update image paths
    for rec in selected:
        rec["image"] = f"images/{Path(str(rec['image'])).name}"

    _write_jsonl(selected, args.out_dir / "figures.jsonl")
    _write_human_means_csv(selected, args.out_dir / "human_means.csv")

    # Manifest from eval1200 consolidated format if available
    manifest_records: list[dict] = []
    if args.source_manifest.exists():
        for rec in _load_jsonl(args.source_manifest):
            key = (str(rec["paper_id"]), int(rec["fig_index"]))
            if key in selected_keys:
                manifest_records.append(rec)
    else:
        manifest_records = [
            {
                "paper_id": r["paper_id"],
                "fig_index": r["fig_index"],
                "venue": r["venue"],
                "year": r["year"],
                "caption": r.get("caption"),
                "context_count": r.get("context_count", 0),
                "annotator_count": r.get("annotator_count", 0),
            }
            for r in selected
        ]

    CONFIGS.mkdir(parents=True, exist_ok=True)
    _write_jsonl(manifest_records, CONFIGS / "human_eval_subset_200.jsonl")

    overlap = len(selected_keys & {(str(r["paper_id"]), int(r["fig_index"])) for r in figures1200})
    meta = {
        "built_at": datetime.now(UTC).isoformat(),
        "method": "stratified-subsample-from-eval1200",
        "source_figures": str(args.source_figures),
        "target_figures": args.n,
        "figure_count": len(selected),
        "missing_images": missing_images,
        "overlap_with_eval1200": overlap,
        "venue_distribution": dict(Counter(r["venue"] for r in selected)),
        "year_distribution": {
            str(k): v for k, v in sorted(Counter(int(r["year"]) for r in selected).items())
        },
        "context_figures": sum(1 for r in selected if int(r.get("context_count") or 0) > 0),
    }
    (args.out_dir / "sampling_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (CONFIGS / "human_eval_subset_200_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    index = f"""SciFigQual VLM Eval Subset (200 figures, stratified subsample from eval1200)

  figures.jsonl     — {len(selected)} records with human means
  human_means.csv   — flat score table
  images/           — PNG files
  sampling_meta.json — selection metadata (overlap_with_eval1200={overlap})

Join key: (paper_id, fig_index) or figure_id
"""
    (args.out_dir / "INDEX.txt").write_text(index, encoding="utf-8")
    print(f"eval200: {len(selected)} figures, missing_images={missing_images}, overlap={overlap}")


if __name__ == "__main__":
    main()
