#!/usr/bin/env python3
"""Aggregate experiment results into paper_experiment_tables.md metrics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code" / "Multidimensional-Assessment-of-Scientific-Paper-Figures-main"
BENCH_SCRIPTS = ROOT / "scripts" / "benchmark"
if str(BENCH_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(BENCH_SCRIPTS))

from manifest_utils import (  # noqa: E402
    human_means_from_consolidated,
    join_model_human,
    load_manifest_df,
    summarize_alignment,
)

import polars as pl  # noqa: E402

TABLE1_RUNS = {
    "D1": ("Direct Judge", "gemini:gemini-2.5-flash", 1),
    "D2": ("Direct Judge", "openai:gpt-4o", 1),
    "D3": ("Direct Judge", "claude:claude-sonnet-4-5", 1),
    "D4": ("Direct Judge", "qwen:qwen-vl-max", 1),
    "D5": ("Direct Judge", "glm:glm-4.6v", 1),
    "D6": ("Direct Judge", "doubao:doubao-seed-2-0-pro-260215", 1),
    "S1": ("Sidecar Judge", "gemini:gemini-2.5-flash+OCR", 1),
    "S2": ("Sidecar Judge", "openai:gpt-4o+OCR", 1),
    "S3": ("Sidecar Judge", "qwen:qwen-vl-max+OCR", 1),
    "S4": ("Sidecar Judge", "glm:glm-4.6v+OCR", 1),
    "S5": ("Sidecar Judge", "doubao:doubao-seed-2-0-pro-260215+OCR", 1),
    "F1": ("SFQ-Agent", "gemini:2.5-flash/flash", 3),
    "F2": ("SFQ-Agent", "gemini:2.5-pro/pro", 3),
    "F3": ("SFQ-Agent", "openai:gpt-4o/gpt-4o", 3),
    "F4": ("SFQ-Agent", "claude/claude", 3),
    "F5": ("SFQ-Agent", "qwen:qwen-vl-max+qwen-plus", 3),
    "F6": ("SFQ-Agent", "glm:glm-4.6v/glm-4.6v", 3),
}

QWEN_OUTPUT_MAP = {
    "D4": [
        CODE_ROOT / "output" / "vlm_qwen-vl-max___llm_none",
        CODE_ROOT / "output" / "direct___eval1200___vlm_qwen-vl-max___llm_none",
    ],
    "S3": [
        CODE_ROOT / "output" / "sidecar___eval1200___vlm_qwen-vl-max___llm_none",
    ],
    "F5": [
        CODE_ROOT / "output" / "vlm_qwen-vl-max___llm_qwen-plus",
        CODE_ROOT / "output" / "agent___eval1200___vlm_qwen-vl-max___llm_qwen-plus",
    ],
}


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _load_leaderboard(path: Path) -> dict | None:
    lb = path / "eval" / "leaderboard.json"
    if lb.exists():
        return json.loads(lb.read_text(encoding="utf-8")).get("summary", {})
    pq = path / "results.parquet"
    return None if not pq.exists() else {"_parquet": str(pq)}


def _metrics_from_parquet(
    parquet: Path,
    *,
    human_df: pl.DataFrame,
    source_name: str | None,
) -> dict:
    model_df = pl.read_parquet(parquet)
    if source_name and "source_name" in model_df.columns:
        merged = join_model_human(model_df, human_df, source_name=source_name)
    else:
        merged = join_model_human(model_df, human_df, source_name=None)
    return summarize_alignment(merged, run_label=source_name or parquet.stem)


def _collect_table1(
    matrix_root: Path,
    manifest: Path,
    human_path: Path,
) -> list[dict]:
    human_all = human_means_from_consolidated(human_path) if human_path.exists() else pl.DataFrame()
    if human_all.is_empty():
        human_all = _human_from_figures_jsonl(ROOT / "datasets" / "eval1200" / "figures.jsonl")
    manifest_df = load_manifest_df(manifest)
    human_df = human_all.join(
        manifest_df.select("paper_id", "fig_index"), on=["paper_id", "fig_index"], how="inner"
    )

    rows: list[dict] = []
    for run_id, (protocol, backend, api_calls) in TABLE1_RUNS.items():
        metrics: dict | None = None
        status = "pending"

        # Matrix output (parquet or jsonl)
        matrix_dir = matrix_root / run_id
        pq = matrix_dir / "results.parquet"
        jsonl = matrix_dir / "results.jsonl"
        if pq.exists():
            plan_path = matrix_root / "experiment_plan.json"
            source = None
            if plan_path.exists():
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                for item in plan.get("runs", []):
                    if item.get("id") == run_id:
                        source = item.get("source_name")
                        break
            metrics = _metrics_from_parquet(pq, human_df=human_df, source_name=source)
            status = "done"
        elif jsonl.exists():
            rows = [
                json.loads(line)
                for line in jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if rows:
                model_df = pl.DataFrame(rows)
                merged = join_model_human(model_df, human_df, source_name=None)
                metrics = summarize_alignment(merged, run_label=run_id)
                status = "done" if metrics.get("overall_n", 0) >= 1000 else "partial"

        # Qwen one-click output
        qwen_dirs = QWEN_OUTPUT_MAP.get(run_id, [])
        if isinstance(qwen_dirs, Path):
            qwen_dirs = [qwen_dirs]
        for qwen_dir in qwen_dirs:
            if not qwen_dir.exists():
                continue
            lb = _load_leaderboard(qwen_dir)
            if lb and "_parquet" not in lb:
                metrics = lb
                status = "done" if metrics.get("overall_n") else "in_progress"
                break
            if (qwen_dir / "results.parquet").exists():
                cfg_path = qwen_dir / "run_config.json"
                source = None
                if cfg_path.exists():
                    source = json.loads(cfg_path.read_text(encoding="utf-8")).get("source_name")
                metrics = _metrics_from_parquet(
                    qwen_dir / "results.parquet",
                    human_df=human_df,
                    source_name=source,
                )
                status = "done" if metrics.get("overall_n") else "in_progress"
                break

        row = {
            "run_id": run_id,
            "protocol": protocol,
            "backend": backend,
            "api_calls": api_calls,
            "status": status,
            "overall_mae": metrics.get("overall_mae") if metrics else None,
            "overall_within1": metrics.get("overall_within1") if metrics else None,
            "overall_srcc": metrics.get("overall_srcc") if metrics else None,
            "overall_bias": metrics.get("overall_bias") if metrics else None,
            "cc_mae": metrics.get("caption_consistency_mae") if metrics else None,
            "ctx_mae": metrics.get("context_consistency_mae") if metrics else None,
            "mr_mae": metrics.get("misleading_risk_mae") if metrics else None,
        }
        rows.append(row)
    return rows


def _human_from_figures_jsonl(path: Path) -> pl.DataFrame:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    rows = []
    for rec in records:
        rows.append({
            "paper_id": str(rec["paper_id"]),
            "fig_index": int(rec["fig_index"]),
            "human_overall_score": rec.get("human_overall_score"),
            "human_visual_clarity": rec.get("human_visual_clarity"),
            "human_structure_layout": rec.get("human_structure_layout"),
            "human_caption_consistency": rec.get("human_caption_consistency"),
            "human_context_consistency": rec.get("human_context_consistency"),
            "human_misleading_risk": rec.get("human_misleading_risk"),
        })
    return pl.DataFrame(rows)


def _update_table1_doc(rows: list[dict], doc_path: Path) -> None:
    if not doc_path.exists():
        return
    text = doc_path.read_text(encoding="utf-8")
    for row in rows:
        run_id = row["run_id"]
        if row.get("overall_mae") is None:
            continue
        w1 = (
            f"{100 * float(row['overall_within1']):.1f}%"
            if row.get("overall_within1") is not None
            else "—"
        )
        best_cls = ' class="best"' if row.get("status") == "done" else ""
        metrics_html = (
            f"<td{best_cls}>{_fmt(row.get('overall_mae'))}</td>"
            f"<td{best_cls}>{w1}</td>"
            f"<td>{_fmt(row.get('overall_srcc'))}</td>"
            f"<td>{_fmt(row.get('overall_bias'))}</td>"
            f"<td>{_fmt(row.get('cc_mae'))}</td>"
            f"<td>{_fmt(row.get('ctx_mae'))}</td>"
            f"<td>{_fmt(row.get('mr_mae'))}</td>"
        )
        # HTML 三线表：按 data-run-id 定位并刷新 7 个指标列
        pattern = (
            rf'(<td data-run-id="{re.escape(run_id)}"[^>]*>{re.escape(run_id)}</td>\s*'
            rf'<td class="backend"[^>]*>[^<]*</td>\s*'
            rf'<td>\d+</td>\s*)'
            rf'(?:<td[^>]*>[^<]*</td>\s*){{7}}'
        )
        text, n = re.subn(pattern, rf"\1{metrics_html}", text, count=1, flags=re.DOTALL)
        if n:
            continue
        # 兼容旧版 Markdown 管道表
        pattern_md = (
            rf"(\| \*\*{re.escape(run_id)}\*\* \| [^|]+\| [^|]+\| )"
            rf"[^|]*( \| )[^|]*( \| )[^|]*( \| )[^|]*( \| )[^|]*( \| )[^|]*( \| )[^|]*( \| )"
        )
        replacement_md = (
            rf"\1{_fmt(row.get('overall_mae'))} | {w1} | {_fmt(row.get('overall_srcc'))} | "
            rf"{_fmt(row.get('overall_bias'))} | {_fmt(row.get('cc_mae'))} | "
            rf"{_fmt(row.get('ctx_mae'))} | {_fmt(row.get('mr_mae'))} |"
        )
        text, _ = re.subn(pattern_md, replacement_md, text, count=1)
    doc_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate experiment tables")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--matrix-root",
        type=Path,
        default=ROOT / "outputs" / "benchmark_eval1200",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "configs" / "human_eval_subset_1200.jsonl",
    )
    parser.add_argument(
        "--human-figures",
        type=Path,
        default=ROOT / "datasets" / "eval1200" / "figures.jsonl",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=ROOT / "docs" / "paper_experiment_tables.md",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.phase == 1:
        rows = _collect_table1(args.matrix_root, args.manifest, args.human_figures)
        out = args.json_out or (ROOT / "outputs" / "table1_metrics.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _update_table1_doc(rows, args.doc)
        print(f"Table 1 metrics: {out}")
        for row in rows:
            print(
                f"  {row['run_id']}: MAE={_fmt(row.get('overall_mae'))} "
                f"status={row['status']}"
            )
    else:
        gen_root = ROOT / "outputs" / "benchmark_eval200"
        out = args.json_out or (ROOT / "outputs" / "table2_metrics.json")
        summary = {"task_root": str(gen_root), "runs": []}
        if gen_root.exists():
            for run_dir in sorted(gen_root.iterdir()):
                if run_dir.is_dir() and (run_dir / "summary.json").exists():
                    summary["runs"].append(json.loads((run_dir / "summary.json").read_text(encoding="utf-8")))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Table 2 metrics: {out} ({len(summary['runs'])} runs)")


if __name__ == "__main__":
    main()
