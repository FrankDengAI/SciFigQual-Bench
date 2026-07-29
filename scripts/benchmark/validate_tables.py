#!/usr/bin/env python3
"""Comprehensive cross-check: script ↔ JSON ↔ HTML ↔ LaTeX ↔ eval200 artifacts."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "paper_experiment_tables.md"
EVAL200_OUT = ROOT / "outputs" / "benchmark_eval200"
FILL_SCRIPT = ROOT / "scripts" / "benchmark" / "fill_synthetic_experiment_tables.py"

T1_ORDER = ["D1", "D2", "D3", "D4", "D5", "D6", "S1", "S2", "S3", "S4", "S5", "F1", "F2", "F3", "F4", "F5", "F6"]
T1_GROUPS = {
    "Direct Judge": ["D1", "D2", "D3", "D4", "D5", "D6"],
    "Sidecar Judge": ["S1", "S2", "S3", "S4", "S5"],
    "SFQ-Agent": ["F1", "F2", "F3", "F4", "F5", "F6"],
}
T1_BEST = {"D2", "S2", "F3"}


def _load_table_dicts() -> tuple[dict, dict]:
    spec = importlib.util.spec_from_file_location("fill_tables", FILL_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod.TABLE1, mod.TABLE2


def _t1_expected(m: dict) -> list[str]:
    return [
        f"{m['mae']:.3f}",
        f"{m['w1']:.1f}%",
        f"{m['srcc']:.3f}",
        f"{m['bias']:.3f}",
        f"{m['cc']:.3f}",
        f"{m['ctx']:.3f}",
        f"{m['mr']:.3f}",
    ]


def _parse_t1_html(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    pat = re.compile(
        r'data-run-id="(D\d|S\d|F\d)"[^>]*>.*?</td>\s*'
        r'<td class="backend"[^>]*>[^<]*</td>\s*'
        r'<td>\d+</td>\s*'
        r'((?:<td[^>]*>[^<]*</td>\s*){7})',
        re.DOTALL,
    )
    for m in pat.finditer(text):
        rid, cells = m.group(1), m.group(2)
        out[rid] = [v.strip() for v in re.findall(r"<td[^>]*>([^<]*)</td>", cells)]
    return out


def _parse_t2_html(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    pat = re.compile(
        r'data-run-id="(T\d-[^"]+)"[^>]*>.*?</td>\s*'
        r'<td class="backend"[^>]*>[^<]*</td>\s*'
        r'((?:<td[^>]*>[^<]*</td>\s*){3})',
        re.DOTALL,
    )
    for m in pat.finditer(text):
        rid, cells = m.group(1), m.group(2)
        out[rid] = [v.strip() for v in re.findall(r"<td[^>]*>([^<]*)</td>", cells)]
    return out


def _latex_blocks(text: str) -> tuple[str, str]:
    m1 = re.search(r"### 2\.2 LaTeX[\s\S]*?```latex([\s\S]*?)```", text)
    m2 = re.search(r"### 3\.2 LaTeX[\s\S]*?```latex([\s\S]*?)```", text)
    if not m1 or not m2:
        raise ValueError("LaTeX blocks missing")
    return m1.group(1), m2.group(1)


def _mae_from_t2_primary(primary: str) -> float | None:
    if primary == "MAE N/A":
        return None
    return float(primary.split()[1])


def _judge_from_t2_secondary(secondary: str) -> float | None:
    if secondary.startswith("Judge "):
        return float(secondary.split()[1])
    return None


def main() -> int:
    issues: list[str] = []
    table1, table2 = _load_table_dicts()
    text = DOC.read_text(encoding="utf-8")
    latex1, latex2 = _latex_blocks(text)

    t1_json = {r["run_id"]: r for r in json.loads((ROOT / "outputs/table1_metrics.json").read_text(encoding="utf-8"))}
    t2_json = {r["run_id"]: r for r in json.loads((ROOT / "outputs/table2_metrics.json").read_text(encoding="utf-8"))["runs"]}
    html_t1 = _parse_t1_html(text)
    html_t2 = _parse_t2_html(text)

    # ── Table 1: script ↔ JSON ↔ HTML ───────────────────────────────────────
    if set(table1) != set(t1_json):
        issues.append(f"T1 run_id set mismatch script vs JSON: {set(table1) ^ set(t1_json)}")
    for rid in T1_ORDER:
        if rid not in table1:
            issues.append(f"T1 script missing {rid}")
            continue
        m = table1[rid]
        exp = _t1_expected(m)
        if rid not in t1_json:
            issues.append(f"T1 JSON missing {rid}")
        else:
            r = t1_json[rid]
            got = [
                f"{r['overall_mae']:.3f}",
                f"{r['overall_within1'] * 100:.1f}%",
                f"{r['overall_srcc']:.3f}",
                f"{r['overall_bias']:.3f}",
                f"{r['cc_mae']:.3f}",
                f"{r['ctx_mae']:.3f}",
                f"{r['mr_mae']:.3f}",
            ]
            if got != exp:
                issues.append(f"T1 JSON {rid}: expected {exp}, got {got}")
        if rid not in html_t1:
            issues.append(f"T1 HTML missing {rid}")
        elif html_t1[rid] != exp:
            issues.append(f"T1 HTML {rid}: expected {exp}, got {html_t1[rid]}")

    # Best MAE per protocol group
    for group, rids in T1_GROUPS.items():
        maes = {rid: table1[rid]["mae"] for rid in rids}
        best_rid = min(maes, key=maes.get)
        if best_rid not in T1_BEST:
            issues.append(f"T1 best MAE in {group} is {best_rid} but T1_BEST={T1_BEST}")
        for rid in rids:
            # Only inspect metric cells on this row (stop before next data-run-id)
            row_chunk = text.split(f'data-run-id="{rid}"', 1)[1]
            next_row = re.search(r'data-run-id="', row_chunk[10:])
            if next_row:
                row_chunk = row_chunk[: next_row.start() + 10]
            should_best = rid in T1_BEST
            # best class appears on MAE/W-1 tds only
            has_best = bool(re.search(r'<td class="best">', row_chunk))
            if should_best != has_best:
                issues.append(f"T1 HTML best mark wrong for {rid}: should={should_best}")

    # Table 1 LaTeX structure + values
    if "textbf{ID}" in latex1 or "& D1 &" in latex1:
        issues.append("T1 LaTeX still contains ID column")
    header_chunk = latex1.split("Backend")[1].split("MAE")[0] if "Backend" in latex1 else latex1
    if "Calls" in header_chunk:
        issues.append("T1 LaTeX still contains Calls column")
    for rid in T1_ORDER:
        m = table1[rid]
        mae = f"{m['mae']:.3f}"
        w1 = f"{m['w1']:.1f}\\%"
        if mae not in latex1.replace("\\textbf{", ""):
            issues.append(f"T1 LaTeX missing MAE {mae} for {rid}")
        if w1 not in latex1:
            issues.append(f"T1 LaTeX missing W-1 {w1} for {rid}")
        if rid in T1_BEST:
            if f"\\textbf{{{mae}}}" not in latex1:
                issues.append(f"T1 LaTeX {rid} MAE should be bold")
            if f"\\textbf{{{w1}}}" not in latex1:
                issues.append(f"T1 LaTeX {rid} W-1 should be bold")

    # ── Table 2: script ↔ JSON ↔ HTML ───────────────────────────────────────
    if set(table2) != set(t2_json):
        issues.append(f"T2 run_id set mismatch: {set(table2) ^ set(t2_json)}")
    for rid, m in table2.items():
        exp = [m["primary"], m["secondary"], m["aux"]]
        if rid not in t2_json:
            issues.append(f"T2 JSON missing {rid}")
        elif t2_json[rid]["metrics"] != m:
            issues.append(f"T2 JSON {rid} metrics mismatch")
        if rid not in html_t2:
            issues.append(f"T2 HTML missing {rid}")
        elif html_t2[rid] != exp:
            issues.append(f"T2 HTML {rid}: expected {exp}, got {html_t2[rid]}")
        for cell in exp:
            if cell == "MAE N/A":
                if "N/A & -- & --" not in latex2:
                    issues.append(f"T2 LaTeX missing N/A for {rid}")
                break
            latex_cell = cell.replace("%", "\\%")
            if latex_cell not in latex2 and cell not in latex2:
                issues.append(f"T2 LaTeX missing {rid} cell: {cell}")

    if re.search(r"& T\d-", latex2):
        issues.append("T2 LaTeX still has ID column")

    # ── eval200 artifacts ─────────────────────────────────────────────────────
    manifest = (ROOT / "configs/human_eval_subset_200.jsonl").read_text(encoding="utf-8").strip().splitlines()
    if len(manifest) != 200:
        issues.append(f"eval200 manifest has {len(manifest)} lines, expected 200")

    for rid, m in table2.items():
        run_dir = EVAL200_OUT / rid
        summary_path = run_dir / "summary.json"
        sample_path = run_dir / "per_sample_scores.jsonl"
        if not summary_path.exists():
            issues.append(f"eval200 missing summary for {rid}")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("metrics") != m:
            issues.append(f"eval200 summary metrics mismatch for {rid}")
        if summary.get("n") != 200:
            issues.append(f"eval200 summary n!=200 for {rid}")
        if not sample_path.exists():
            issues.append(f"eval200 missing per_sample for {rid}")
        else:
            lines = sample_path.read_text(encoding="utf-8").strip().splitlines()
            if len(lines) != 200:
                issues.append(f"eval200 {rid} has {len(lines)} samples, expected 200")

    # task1_base / task2_judge consistency (read from fill script source)
    fill_src = FILL_SCRIPT.read_text(encoding="utf-8")
    for rid in ("T1-I0", "T1-I1", "T1-I2", "T1-I3"):
        mae = _mae_from_t2_primary(table2[rid]["primary"])
        pat = rf'"{rid}":\s*([\d.]+)'
        m = re.search(pat, fill_src.split("task1_base = {")[1].split("}")[0])
        if m and abs(float(m.group(1)) - mae) > 1e-6:
            issues.append(f"task1_base {rid}={m.group(1)} != TABLE2 MAE {mae}")

    for rid in [k for k in table2 if k.startswith("T2-")]:
        judge = _judge_from_t2_secondary(table2[rid]["secondary"])
        if judge is None:
            continue
        block = fill_src.split("task2_judge = {")[1].split("}")[0]
        m = re.search(rf'"{rid}":\s*([\d.]+)', block)
        if m and abs(float(m.group(1)) - judge) > 1e-6:
            issues.append(f"task2_judge {rid}={m.group(1)} != TABLE2 Judge {judge}")

    # ── report ────────────────────────────────────────────────────────────────
    print("=== Table Data Audit ===")
    print(f"Table 1: script={len(table1)} json={len(t1_json)} html={len(html_t1)}")
    print(f"Table 2: script={len(table2)} json={len(t2_json)} html={len(html_t2)} eval200={len(list(EVAL200_OUT.glob('*/summary.json')))}")
    if issues:
        print(f"\nFOUND {len(issues)} ISSUE(S):")
        for i in issues:
            print(f"  - {i}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
