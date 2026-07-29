#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["polars", "pillow"]
# ///
"""Build self-contained datasets/ folder from human_annotations_consolidated.jsonl.

Outputs:
  datasets/full/       — all ~6355 annotated figures (images/ + JSONL/CSV text)
  datasets/eval1200/   — deterministic VLM eval subset (1200 figures)
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import polars as pl
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VIEWABLE = ROOT / "ccf-paper-figures-viewable"
PARQUET_ROOT = ROOT / "ccf-paper-figures" / "processed"
DEFAULT_CONSOLIDATED = VIEWABLE / "human_annotations_consolidated.jsonl"
DATASETS = ROOT / "datasets"

sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))
from manifest_utils import DIMENSIONS, load_consolidated, write_manifest_jsonl  # noqa: E402
from sample_human_eval_subset import build_meta, sample_subset  # noqa: E402


def _figure_id(paper_id: str, fig_index: int) -> str:
    return f"{paper_id}__fig{int(fig_index):04d}"


def _image_name(paper_id: str, fig_index: int) -> str:
    return f"{_figure_id(paper_id, fig_index)}.png"


def _human_means(annotations: list[dict[str, Any]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for dim in DIMENSIONS + ["overall_score"]:
        vals = [a[dim] for a in annotations if a.get(dim) is not None]
        out[f"human_{dim}"] = round(sum(vals) / len(vals), 4) if vals else None
    return out


def _serialize_record(rec: dict[str, Any], *, image_rel: str) -> dict[str, Any]:
    paper_id = str(rec["paper_id"])
    fig_index = int(rec["fig_index"])
    means = _human_means(rec.get("annotations") or [])
    return {
        "figure_id": _figure_id(paper_id, fig_index),
        "paper_id": paper_id,
        "paper_id_norm": rec.get("paper_id_norm"),
        "fig_index": fig_index,
        "venue": rec.get("venue"),
        "year": int(rec["year"]),
        "title": rec.get("title"),
        "caption": rec.get("caption"),
        "section": rec.get("section"),
        "context_texts": rec.get("context_texts") or [],
        "context_count": int(rec.get("context_count") or 0),
        "domain_l1": rec.get("domain_l1"),
        "domain_l2": rec.get("domain_l2"),
        "width": rec.get("width"),
        "height": rec.get("height"),
        "annotator_count": int(rec.get("annotator_count") or 0),
        "image": image_rel,
        **means,
        "annotations": rec.get("annotations") or [],
    }


def _record_exportable(
    rec: dict[str, Any],
    *,
    cache: dict[tuple[str, int], pl.DataFrame],
) -> bool:
    src = VIEWABLE / str(rec.get("image_file") or "")
    if src.exists() and src.stat().st_size > 0:
        return True
    venue = str(rec["venue"])
    year = int(rec["year"])
    paper_id = str(rec["paper_id"])
    fig_index = int(rec["fig_index"])
    key = (venue, year)
    if key not in cache:
        shard = _load_parquet_shard(venue, year)
        cache[key] = shard if shard is not None else pl.DataFrame()
    shard = cache[key]
    if shard.is_empty():
        return False
    rows = shard.filter(
        (pl.col("paper_id") == paper_id) & (pl.col("fig_index") == fig_index)
    )
    if rows.is_empty():
        return False
    raw = rows["image_bytes"][0]
    return bool(raw)


def _filter_exportable(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cache: dict[tuple[str, int], pl.DataFrame] = {}
    ok: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for rec in records:
        if _record_exportable(rec, cache=cache):
            ok.append(rec)
        else:
            bad.append(rec)
    return ok, bad


def _load_parquet_shard(venue: str, year: int) -> pl.DataFrame | None:
    path = PARQUET_ROOT / venue / str(year) / "figures_clean.parquet"
    if not path.exists():
        return None
    return pl.read_parquet(path)


def _parquet_lookup_cache() -> dict[tuple[str, int], pl.DataFrame]:
    return {}


def _export_png_from_parquet(
    *,
    venue: str,
    year: int,
    paper_id: str,
    fig_index: int,
    dest: Path,
    cache: dict[tuple[str, int], pl.DataFrame],
) -> bool:
    key = (venue, year)
    if key not in cache:
        shard = _load_parquet_shard(venue, year)
        cache[key] = shard if shard is not None else pl.DataFrame()
    shard = cache[key]
    if shard.is_empty():
        return False
    rows = shard.filter(
        (pl.col("paper_id") == paper_id) & (pl.col("fig_index") == fig_index)
    )
    if rows.is_empty():
        return False
    raw = rows["image_bytes"][0]
    if not raw:
        return False
    try:
        img = Image.open(BytesIO(bytes(raw)))
        if img.mode not in ("RGB", "RGBA", "L", "LA"):
            img = img.convert("RGB")
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, format="PNG", optimize=True)
        return True
    except Exception:
        return False


def _resolve_image(
    rec: dict[str, Any],
    *,
    images_dir: Path,
    cache: dict[tuple[str, int], pl.DataFrame],
    skip_existing: bool,
    full_images_dir: Path | None = None,
) -> tuple[Path, str]:
    paper_id = str(rec["paper_id"])
    fig_index = int(rec["fig_index"])
    name = _image_name(paper_id, fig_index)
    dest = images_dir / name
    rel = f"images/{name}"

    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        return dest, rel

    if full_images_dir is not None:
        cached = full_images_dir / name
        if cached.exists() and cached.stat().st_size > 0:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, dest)
            return dest, rel

    src = VIEWABLE / str(rec.get("image_file") or "")
    if src.exists() and src.stat().st_size > 0:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest, rel

    ok = _export_png_from_parquet(
        venue=str(rec["venue"]),
        year=int(rec["year"]),
        paper_id=paper_id,
        fig_index=fig_index,
        dest=dest,
        cache=cache,
    )
    if not ok:
        raise FileNotFoundError(
            f"Could not resolve image for {paper_id} fig{fig_index} "
            f"(viewable={rec.get('image_file')})"
        )
    return dest, rel


def _write_figures_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_human_means_csv(records: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "figure_id",
        "paper_id",
        "fig_index",
        "venue",
        "year",
        "image",
        "caption",
        "context_count",
        "annotator_count",
        *[f"human_{d}" for d in DIMENSIONS],
        "human_overall_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k) for k in fields}
            writer.writerow(row)


def _write_index(path: Path, *, title: str, figure_count: int, extra: str = "") -> None:
    text = f"""{title}
Built: {datetime.now(UTC).isoformat()}

Figure count: {figure_count}

Files:
  images/           — PNG files, one per figure
  figures.jsonl     — full metadata, caption, context, raw human annotations
  human_means.csv   — flat table of human-mean scores (open in Excel)
{extra}
Join key: (paper_id, fig_index) or figure_id

Human gold standard:
  Mean over all annotators in annotations[] for each dimension and overall_score.
"""
    path.write_text(text.strip() + "\n", encoding="utf-8")


def export_bundle(
    records: list[dict[str, Any]],
    *,
    out_dir: Path,
    skip_existing: bool,
    full_images_dir: Path | None = None,
) -> dict[str, Any]:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, int], pl.DataFrame] = {}

    exported: list[dict[str, Any]] = []
    missing: list[str] = []
    for index, rec in enumerate(records, start=1):
        try:
            _, rel = _resolve_image(
                rec,
                images_dir=images_dir,
                cache=cache,
                skip_existing=skip_existing,
                full_images_dir=full_images_dir,
            )
            exported.append(_serialize_record(rec, image_rel=rel))
        except FileNotFoundError as exc:
            missing.append(str(exc))
        if index % 500 == 0:
            print(f"  images {index}/{len(records)}")

    if missing:
        report = out_dir / "missing_images.txt"
        report.write_text("\n".join(missing) + "\n", encoding="utf-8")

    exported.sort(key=lambda r: (r["venue"], r["year"], r["paper_id"], r["fig_index"]))
    _write_figures_jsonl(exported, out_dir / "figures.jsonl")
    _write_human_means_csv(exported, out_dir / "human_means.csv")
    return {
        "figure_count": len(exported),
        "missing_images": len(missing),
        "images_dir": str(images_dir),
    }


def _prune_orphan_images(out_dir: Path) -> None:
    figures_path = out_dir / "figures.jsonl"
    if not figures_path.exists():
        return
    needed = {
        json.loads(line)["image"].split("/")[-1]
        for line in figures_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    images_dir = out_dir / "images"
    for path in images_dir.glob("*.png"):
        if path.name not in needed:
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build datasets/full and eval subsets")
    parser.add_argument("--input", type=Path, default=DEFAULT_CONSOLIDATED)
    parser.add_argument("--datasets-root", type=Path, default=DATASETS)
    parser.add_argument(
        "--eval-size",
        type=int,
        default=None,
        help="Single eval subset size (legacy; use --eval-sizes for multiple)",
    )
    parser.add_argument(
        "--eval-sizes",
        type=str,
        default="1200,200",
        help="Comma-separated eval subset sizes, e.g. 1200,200",
    )
    parser.add_argument("--skip-existing-images", action="store_true", default=True)
    parser.add_argument("--force-images", action="store_true", help="Re-copy all images")
    args = parser.parse_args()

    if args.eval_size is not None:
        eval_sizes = [args.eval_size]
    else:
        eval_sizes = [int(x.strip()) for x in args.eval_sizes.split(",") if x.strip()]

    skip_existing = args.skip_existing_images and not args.force_images
    records = load_consolidated(args.input)
    records.sort(key=lambda r: (r["venue"], r["year"], r["paper_id"], r["fig_index"]))
    print(f"Loaded {len(records)} records from {args.input}")

    exportable, excluded = _filter_exportable(records)
    print(f"Exportable figures: {len(exportable)} / {len(records)} ({len(excluded)} excluded — no image source)")

    full_dir = args.datasets_root / "full"

    if excluded:
        write_manifest_jsonl(excluded, args.datasets_root / "excluded_no_image.jsonl")

    print("\n=== Export datasets/full ===")
    full_stats = export_bundle(exportable, out_dir=full_dir, skip_existing=skip_existing)
    _write_index(
        full_dir / "INDEX.txt",
        title="SciFigQual Full Dataset (all exportable human-annotated figures)",
        figure_count=full_stats["figure_count"],
        extra=(
            f"  ../excluded_no_image.jsonl — {len(excluded)} records in source with no PNG/parquet image\n"
            if excluded
            else ""
        ),
    )
    print(f"Full: {full_stats['figure_count']} figures, missing={full_stats['missing_images']}")

    configs_dir = ROOT / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    eval_stats_map: dict[str, Any] = {}
    eval_sampling_map: dict[str, Any] = {}

    for eval_size in sorted(eval_sizes, reverse=True):
        eval_name = f"eval{eval_size}"
        eval_dir = args.datasets_root / eval_name
        manifest_name = f"human_eval_subset_{eval_size}.jsonl"
        meta_name = f"human_eval_subset_{eval_size}_meta.json"

        print(f"\n=== Sample {eval_name} (from exportable pool only) ===")
        eval_records = sample_subset(exportable, n_figures=eval_size)
        meta = build_meta(eval_records, source=args.input)
        meta["target_figures"] = eval_size
        meta["output_dir"] = str(eval_dir)
        meta["eval_name"] = eval_name

        write_manifest_jsonl(eval_records, configs_dir / manifest_name)
        (configs_dir / meta_name).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print(f"\n=== Export datasets/{eval_name} ===")
        stats = export_bundle(
            eval_records,
            out_dir=eval_dir,
            skip_existing=skip_existing,
            full_images_dir=full_dir / "images",
        )
        (eval_dir / "sampling_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_index(
            eval_dir / "INDEX.txt",
            title=f"SciFigQual VLM Eval Subset ({eval_size} figures, deterministic sample)",
            figure_count=stats["figure_count"],
            extra="  sampling_meta.json — how this subset was selected\n",
        )
        _prune_orphan_images(eval_dir)
        print(f"{eval_name}: {stats['figure_count']} figures, missing={stats['missing_images']}")

        eval_stats_map[eval_name] = stats
        eval_sampling_map[eval_name] = {
            "figure_count": len(eval_records),
            "paper_count": len({r["paper_id"] for r in eval_records}),
            "venue_distribution": dict(Counter(r["venue"] for r in eval_records)),
            "year_distribution": {
                str(k): v
                for k, v in sorted(Counter(int(r["year"]) for r in eval_records).items())
            },
            "overlap_with_eval1200": meta.get("overlap_with_eval1200"),
        }

    # Record overlap of eval200 with eval1200 if both exist
    if 1200 in eval_sizes and 200 in eval_sizes:
        eval1200_path = configs_dir / "human_eval_subset_1200.jsonl"
        eval200_path = configs_dir / "human_eval_subset_200.jsonl"
        if eval1200_path.exists() and eval200_path.exists():
            keys1200 = {
                (str(r["paper_id"]), int(r["fig_index"]))
                for r in load_consolidated(eval1200_path)
            }
            rec200 = load_consolidated(eval200_path)
            overlap200 = sum(
                1
                for r in rec200
                if (str(r["paper_id"]), int(r["fig_index"])) in keys1200
            )
            for path in [
                args.datasets_root / "eval200" / "sampling_meta.json",
                configs_dir / "human_eval_subset_200_meta.json",
            ]:
                if path.exists():
                    meta200 = json.loads(path.read_text(encoding="utf-8"))
                    meta200["overlap_with_eval1200"] = overlap200
                    path.write_text(
                        json.dumps(meta200, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
            eval_sampling_map["eval200"]["overlap_with_eval1200"] = overlap200
            print(f"\neval200 overlap with eval1200: {overlap200}/200")

    summary = {
        "built_at": datetime.now(UTC).isoformat(),
        "source": str(args.input),
        "source_records": len(records),
        "exportable_records": len(exportable),
        "excluded_no_image": len(excluded),
        "full": full_stats,
        "eval_subsets": eval_stats_map,
        "eval_sampling": eval_sampling_map,
    }
    (args.datasets_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    import subprocess
    stats_script = ROOT / "scripts" / "compute_corpus_stats.py"
    subprocess.run([sys.executable, str(stats_script)], check=True)
    print(f"\nDone. See {args.datasets_root}")


if __name__ == "__main__":
    main()
