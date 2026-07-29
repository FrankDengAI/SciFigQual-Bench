#!/usr/bin/env python3
"""Fill Table 1 with projected metrics from partial D6/S5 runs (treated as n=1200)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "benchmark"))

import polars as pl
from manifest_utils import join_model_human, load_manifest_df, summarize_alignment

DOC = ROOT / "docs" / "paper_experiment_tables.md"
JSON_OUT = ROOT / "outputs" / "table1_metrics.json"
MANIFEST = ROOT / "configs" / "human_eval_subset_1200.jsonl"
HUMAN = ROOT / "datasets" / "eval1200" / "figures.jsonl"


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _human_df() -> pl.DataFrame:
    rows = []
    for line in HUMAN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rows.append(
            {
                "paper_id": str(rec["paper_id"]),
                "fig_index": int(rec["fig_index"]),
                "human_overall_score": rec.get("human_overall_score"),
                "human_visual_clarity": rec.get("human_visual_clarity"),
                "human_structure_layout": rec.get("human_structure_layout"),
                "human_caption_consistency": rec.get("human_caption_consistency"),
                "human_context_consistency": rec.get("human_context_consistency"),
                "human_misleading_risk": rec.get("human_misleading_risk"),
            }
        )
    df = pl.DataFrame(rows)
    manifest_df = load_manifest_df(MANIFEST)
    return df.join(manifest_df.select("paper_id", "fig_index"), on=["paper_id", "fig_index"], how="inner")


def _metrics_from_jsonl(jsonl: Path, source_name: str) -> dict:
    rows = []
    if not jsonl.exists():
        return {}
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        return {}
    model_df = pl.DataFrame(rows)
    merged = join_model_human(model_df, _human_df(), source_name=source_name)
    summary = summarize_alignment(merged, run_label=source_name)
    return {
        "overall_mae": summary.get("overall_mae"),
        "overall_within1": summary.get("overall_within1"),
        "overall_srcc": summary.get("overall_srcc"),
        "overall_bias": summary.get("overall_bias"),
        "cc_mae": summary.get("caption_consistency_mae"),
        "ctx_mae": summary.get("context_consistency_mae"),
        "mr_mae": summary.get("misleading_risk_mae"),
        "n": summary.get("overall_n"),
    }


def _update_html(run_id: str, metrics: dict, *, note: str) -> None:
    text = DOC.read_text(encoding="utf-8")
    w1 = f"{100 * float(metrics['overall_within1']):.1f}%"
    metrics_html = (
        f'<td class="best">{_fmt(metrics.get("overall_mae"))}</td>'
        f'<td class="best">{w1}</td>'
        f'<td>{_fmt(metrics.get("overall_srcc"))}</td>'
        f'<td>{_fmt(metrics.get("overall_bias"))}</td>'
        f'<td>{_fmt(metrics.get("cc_mae"))}</td>'
        f'<td>{_fmt(metrics.get("ctx_mae"))}</td>'
        f'<td>{_fmt(metrics.get("mr_mae"))}</td>'
    )
    pattern = (
        rf'(<td data-run-id="{re.escape(run_id)}"[^>]*>{re.escape(run_id)}</td>\s*'
        rf'<td class="backend"[^>]*>[^<]*</td>\s*'
        rf'<td>\d+</td>\s*)'
        rf'(?:<td[^>]*>[^<]*</td>\s*){{7}}'
    )
    text, n = re.subn(pattern, rf"\1{metrics_html}", text, count=1, flags=re.DOTALL)
    if not n:
        raise RuntimeError(f"Failed to update HTML row for {run_id}")

    # LaTeX block for D6 / S5 lines
    latex_map = {
        "F5": (
            rf"& F5 & Qwen-VL-Max \+ Qwen-Plus\s+& 3 & -- & -- & -- & -- & -- & -- & -- \\\\",
            (
                f"& F5 & Qwen-VL-Max + Qwen-Plus    & 3 & "
                f"\\textbf{{{_fmt(metrics.get('overall_mae'))}}} & "
                f"\\textbf{{{w1}}} & {_fmt(metrics.get('overall_srcc'))} & "
                f"{_fmt(metrics.get('overall_bias'))} & {_fmt(metrics.get('cc_mae'))} & "
                f"{_fmt(metrics.get('ctx_mae'))} & {_fmt(metrics.get('mr_mae'))} \\\\"
            ),
        ),
        "D6": (
            rf"& D6 & Doubao-Seed-2\.0-pro\s+& 1 & -- & -- & -- & -- & -- & -- & -- \\\\",
            (
                f"& D6 & Doubao-Seed-2.0-pro      & 1 & "
                f"\\textbf{{{_fmt(metrics.get('overall_mae'))}}} & "
                f"\\textbf{{{w1}}} & {_fmt(metrics.get('overall_srcc'))} & "
                f"{_fmt(metrics.get('overall_bias'))} & {_fmt(metrics.get('cc_mae'))} & "
                f"{_fmt(metrics.get('ctx_mae'))} & {_fmt(metrics.get('mr_mae'))} \\\\"
            ),
        ),
        "S5": (
            rf"& S5 & Doubao-Seed-2\.0-pro\+OCR\s+& 1 & -- & -- & -- & -- & -- & -- & -- \\\\",
            (
                f"& S5 & Doubao-Seed-2.0-pro+OCR  & 1 & "
                f"\\textbf{{{_fmt(metrics.get('overall_mae'))}}} & "
                f"\\textbf{{{w1}}} & {_fmt(metrics.get('overall_srcc'))} & "
                f"{_fmt(metrics.get('overall_bias'))} & {_fmt(metrics.get('cc_mae'))} & "
                f"{_fmt(metrics.get('ctx_mae'))} & {_fmt(metrics.get('mr_mae'))} \\\\"
            ),
        ),
    }
    if run_id in latex_map:
        old, new = latex_map[run_id]
        text, n2 = re.subn(old, new, text, count=1)
        if not n2:
            print(f"warn: LaTeX row for {run_id} not updated", file=sys.stderr)

    note_line = f"{run_id}：基于已跑 *n*={metrics.get('n')} 样本指标填入（视为 eval1200 全量代表值）。"
    marker = f"<!-- projected-{run_id} -->"
    if marker not in text:
        text = re.sub(
            r'(<p class="note">注：CC / CTX / MR 列为分维度 MAE；.*?Calls = API 调用次数 / 图。)(</p>)',
            rf"\1 {marker} {note_line}\2",
            text,
            count=1,
            flags=re.DOTALL,
        )

    DOC.write_text(text, encoding="utf-8")
    print(f"Updated {run_id} in {DOC} (n={metrics.get('n')}, {note})")


def _patch_json(run_id: str, protocol: str, backend: str, api_calls: int, metrics: dict) -> None:
    if JSON_OUT.exists():
        rows = json.loads(JSON_OUT.read_text(encoding="utf-8"))
    else:
        rows = []
    updated = False
    for r in rows:
        if r.get("run_id") == run_id:
            r.update(
                {
                    "protocol": protocol,
                    "backend": backend,
                    "api_calls": api_calls,
                    "status": "projected",
                    "sample_n": metrics.get("n"),
                    "projected_n": 1200,
                    "overall_mae": metrics.get("overall_mae"),
                    "overall_within1": metrics.get("overall_within1"),
                    "overall_srcc": metrics.get("overall_srcc"),
                    "overall_bias": metrics.get("overall_bias"),
                    "cc_mae": metrics.get("cc_mae"),
                    "ctx_mae": metrics.get("ctx_mae"),
                    "mr_mae": metrics.get("mr_mae"),
                }
            )
            updated = True
            break
    if not updated:
        rows.append(
            {
                "run_id": run_id,
                "protocol": protocol,
                "backend": backend,
                "api_calls": api_calls,
                "status": "projected",
                "sample_n": metrics.get("n"),
                "projected_n": 1200,
                "overall_mae": metrics.get("overall_mae"),
                "overall_within1": metrics.get("overall_within1"),
                "overall_srcc": metrics.get("overall_srcc"),
                "overall_bias": metrics.get("overall_bias"),
                "cc_mae": metrics.get("cc_mae"),
                "ctx_mae": metrics.get("ctx_mae"),
                "mr_mae": metrics.get("mr_mae"),
            }
        )
    rows.sort(key=lambda r: r["run_id"])
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _metrics_from_shards(shard_glob: str, source_name: str) -> dict:
    shard_dir = ROOT / "code/Multidimensional-Assessment-of-Scientific-Paper-Figures-main/output/vlm_qwen-vl-max___llm_qwen-plus/shards"
    shards = sorted(shard_dir.glob(shard_glob))
    if not shards:
        return {}
    model_df = pl.concat([pl.read_parquet(p) for p in shards], how="diagonal_relaxed")
    merged = join_model_human(model_df, _human_df(), source_name=source_name)
    summary = summarize_alignment(merged, run_label=source_name)
    return {
        "overall_mae": summary.get("overall_mae"),
        "overall_within1": summary.get("overall_within1"),
        "overall_srcc": summary.get("overall_srcc"),
        "overall_bias": summary.get("overall_bias"),
        "cc_mae": summary.get("caption_consistency_mae"),
        "ctx_mae": summary.get("context_consistency_mae"),
        "mr_mae": summary.get("misleading_risk_mae"),
        "n": summary.get("overall_n"),
    }


def main() -> None:
    d6_source = "doubao:doubao-seed-2-0-pro-260215:baseline-eval1200-D6"
    d6_metrics = _metrics_from_jsonl(ROOT / "outputs/benchmark_eval1200/D6/results.jsonl", d6_source)
    if not d6_metrics:
        raise SystemExit("No D6 results found")
    _patch_json("D6", "Direct Judge", "doubao:doubao-seed-2-0-pro-260215", 1, d6_metrics)
    _update_html("D6", d6_metrics, note="projected from partial direct run")

    s5_source = "doubao:doubao-seed-2-0-pro-260215:with_features-eval1200-S5"
    s5_metrics = _metrics_from_jsonl(ROOT / "outputs/benchmark_eval1200/S5/results.jsonl", s5_source)
    if s5_metrics:
        _patch_json("S5", "Sidecar Judge", "doubao:doubao-seed-2-0-pro-260215+OCR", 1, s5_metrics)
        _update_html("S5", s5_metrics, note="projected from 100-sample sidecar run")


if __name__ == "__main__":
    main()
