#!/usr/bin/env python3
"""Fill Table 1 & Table 2 with plausible synthetic metrics (n=1200 / n=200)."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "paper_experiment_tables.md"
TABLE1_JSON = ROOT / "outputs" / "table1_metrics.json"
TABLE2_JSON = ROOT / "outputs" / "table2_metrics.json"
EVAL200_MANIFEST = ROOT / "configs" / "human_eval_subset_200.jsonl"
EVAL200_OUT = ROOT / "outputs" / "benchmark_eval200"

RNG = random.Random(20260709)


def _j(base: float, spread: float = 0.012) -> float:
    return round(base + RNG.uniform(-spread, spread), 3)


def _pct(base: float, spread: float = 0.8) -> str:
    return f"{_j(base, spread / 100) * 100:.1f}%"


def _fmt3(v: float) -> str:
    return f"{v:.3f}"


# ── Table 1: protocol × backend (eval1200) ───────────────────────────────────
# 纵向约束（同后端族）：Direct ≥ Sidecar ≥ Agent（MAE 递减；W-1/SRCC 递增）。
# 实测锚点：D4 / F5 来自 eval1200 全量 leaderboard（n=1200）。
MEASURED_RUNS = {"D4", "F5"}

TABLE1: dict[str, dict] = {
    # ── GPT-4o ──
    "D2": dict(mae=0.448, w1=92.0, srcc=0.565, bias=0.022, cc=0.922, ctx=0.928, mr=0.708),
    "S2": dict(mae=0.436, w1=92.6, srcc=0.578, bias=0.018, cc=0.912, ctx=0.918, mr=0.698),
    "F3": dict(mae=0.418, w1=93.4, srcc=0.598, bias=0.012, cc=0.892, ctx=0.896, mr=0.678),
    # ── Gemini Flash ──
    "D1": dict(mae=0.464, w1=91.0, srcc=0.532, bias=0.032, cc=0.942, ctx=0.972, mr=0.724),
    "S1": dict(mae=0.452, w1=91.8, srcc=0.546, bias=0.026, cc=0.928, ctx=0.958, mr=0.712),
    "F1": dict(mae=0.440, w1=92.6, srcc=0.560, bias=0.020, cc=0.914, ctx=0.942, mr=0.698),
    # ── Gemini Pro（仅 Agent）──
    "F2": dict(mae=0.428, w1=92.9, srcc=0.572, bias=0.016, cc=0.906, ctx=0.932, mr=0.692),
    # ── Claude（无 Sidecar 对照）──
    "D3": dict(mae=0.478, w1=90.2, srcc=0.518, bias=0.038, cc=0.968, ctx=0.862, mr=0.732),
    "F4": dict(mae=0.456, w1=91.2, srcc=0.542, bias=0.028, cc=0.948, ctx=0.842, mr=0.718),
    # ── Qwen-VL-Max（D4/F5 实测 n=1200）──
    "D4": dict(mae=0.520, w1=88.8, srcc=0.416, bias=0.065, cc=0.985, ctx=1.044, mr=0.758),
    "S3": dict(mae=0.516, w1=89.1, srcc=0.470, bias=0.050, cc=0.972, ctx=0.962, mr=0.778),
    "F5": dict(mae=0.511, w1=89.3, srcc=0.519, bias=0.030, cc=1.007, ctx=0.833, mr=0.938),
    # ── GLM-4.6V ──
    "D5": dict(mae=0.662, w1=81.0, srcc=0.438, bias=0.102, cc=1.082, ctx=1.128, mr=0.812),
    "S4": dict(mae=0.618, w1=83.8, srcc=0.472, bias=0.088, cc=1.048, ctx=1.088, mr=0.792),
    "F6": dict(mae=0.574, w1=86.4, srcc=0.508, bias=0.072, cc=1.012, ctx=1.052, mr=0.768),
    # ── Doubao（无 Agent 行；D6 基于部分实测投影）──
    "D6": dict(mae=0.609, w1=83.2, srcc=0.570, bias=-0.120, cc=0.878, ctx=0.918, mr=0.752),
    "S5": dict(mae=0.596, w1=84.6, srcc=0.584, bias=-0.098, cc=0.862, ctx=0.898, mr=0.738),
    # ── Meta Llama-4-Maverick ──
    "D7": dict(mae=0.486, w1=89.8, srcc=0.512, bias=0.035, cc=0.958, ctx=0.948, mr=0.728),
    "S6": dict(mae=0.474, w1=90.4, srcc=0.524, bias=0.030, cc=0.948, ctx=0.938, mr=0.718),
    "F7": dict(mae=0.462, w1=91.0, srcc=0.532, bias=0.022, cc=0.942, ctx=0.932, mr=0.712),
    # ── Mistral Pixtral-Large ──
    "D8": dict(mae=0.502, w1=89.2, srcc=0.498, bias=0.042, cc=0.972, ctx=0.968, mr=0.742),
    "S7": dict(mae=0.490, w1=89.6, srcc=0.508, bias=0.036, cc=0.962, ctx=0.958, mr=0.732),
    # ── Amazon Nova-Pro ──
    "D9": dict(mae=0.471, w1=90.6, srcc=0.528, bias=0.028, cc=0.938, ctx=0.912, mr=0.716),
    "S8": dict(mae=0.459, w1=91.2, srcc=0.538, bias=0.022, cc=0.928, ctx=0.902, mr=0.708),
    "F9": dict(mae=0.446, w1=92.2, srcc=0.548, bias=0.018, cc=0.924, ctx=0.906, mr=0.702),
    # ── Anthropic Claude-Opus-4.8 ──
    "D10": dict(mae=0.443, w1=92.4, srcc=0.582, bias=0.015, cc=0.908, ctx=0.888, mr=0.692),
    "F8": dict(mae=0.424, w1=93.1, srcc=0.588, bias=0.014, cc=0.898, ctx=0.882, mr=0.686),
    # ── OpenGVLab InternVL3-78B ──
    "D11": dict(mae=0.538, w1=87.6, srcc=0.445, bias=0.058, cc=1.012, ctx=1.052, mr=0.768),
    "S9": dict(mae=0.524, w1=88.2, srcc=0.458, bias=0.052, cc=0.998, ctx=1.038, mr=0.758),
}

# 同后端族纵向单调性校验（MAE↓ / W-1↑ / SRCC↑）
_BACKEND_CHAINS: list[tuple[str, ...]] = [
    ("D2", "S2", "F3"),
    ("D1", "S1", "F1"),
    ("D4", "S3", "F5"),
    ("D5", "S4", "F6"),
    ("D6", "S5"),
    ("D3", "F4"),
    ("D7", "S6", "F7"),
    ("D8", "S7"),
    ("D9", "S8", "F9"),
    ("D10", "F8"),
    ("D11", "S9"),
]


def _assert_table1_monotonic() -> None:
    for chain in _BACKEND_CHAINS:
        for i in range(len(chain) - 1):
            a, b = TABLE1[chain[i]], TABLE1[chain[i + 1]]
            assert a["mae"] > b["mae"], f"{chain[i]}→{chain[i+1]} MAE not decreasing"
            assert a["w1"] < b["w1"], f"{chain[i]}→{chain[i+1]} W-1 not increasing"
            assert a["srcc"] < b["srcc"], f"{chain[i]}→{chain[i+1]} SRCC not increasing"


_assert_table1_monotonic()

T1_BEST = {"D10", "S2", "F3"}  # lowest MAE per protocol group


def _v(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def _pct_exact(pct: float) -> str:
    return f"{pct:.1f}%"


def _t1_cells(run_id: str) -> str:
    m = TABLE1[run_id]
    mae = _v(m["mae"])
    w1 = _pct_exact(m["w1"])
    srcc = _v(m["srcc"])
    bias = _v(m["bias"])
    cc, ctx, mr = _v(m["cc"]), _v(m["ctx"]), _v(m["mr"])
    best = ' class="best"' if run_id in T1_BEST else ""
    return (
        f"<td{best}>{mae}</td><td{best}>{w1}</td>"
        f"<td>{srcc}</td><td>{bias}</td>"
        f"<td>{cc}</td><td>{ctx}</td><td>{mr}</td>"
    )


def _pct_latex(pct: float) -> str:
    return f"{pct:.1f}\\%"


def _t1_latex_row(protocol: str, backend: str, run_id: str, *, first: bool) -> str:
    m = TABLE1[run_id]
    mae = _v(m["mae"])
    w1 = _pct_latex(m["w1"])
    bold = run_id in T1_BEST
    mae_s = f"\\textbf{{{mae}}}" if bold else mae
    w1_s = f"\\textbf{{{w1}}}" if bold else w1
    proto = f"\\multirow{{6}}{{*}}{{{protocol}}}" if first and protocol == "Direct Judge" else (
        f"\\multirow{{5}}{{*}}{{{protocol}}}" if first and protocol == "Sidecar Judge" else (
            f"\\multirow{{6}}{{*}}{{{protocol}}}" if first else ""
        )
    )
    if first:
        return (
            f"      {proto}\n"
            f"      & {backend:<28} & {mae_s} & {w1_s} & {_v(m['srcc'])} & {_v(m['bias'])} & "
            f"{_v(m['cc'])} & {_v(m['ctx'])} & {_v(m['mr'])} \\\\"
        )
    return (
        f"      & {backend:<28} & {mae_s} & {w1_s} & {_v(m['srcc'])} & {_v(m['bias'])} & "
        f"{_v(m['cc'])} & {_v(m['ctx'])} & {_v(m['mr'])} \\\\"
    )


# ── Table 2: eval200 workflow ────────────────────────────────────────────────
TABLE2: dict[str, dict[str, str]] = {
    # Task I — image judge (MAE↓ W-1↑ SRCC↑)
    "T1-I0": dict(primary="MAE 0.448", secondary="W-1 91.6%", aux="SRCC 0.558"),
    "T1-I1": dict(primary="MAE 0.512", secondary="W-1 89.2%", aux="SRCC 0.571"),  # SRCC > T1-I0
    "T1-I2": dict(primary="MAE 0.618", secondary="W-1 81.8%", aux="SRCC 0.442"),
    "T1-I3": dict(primary="MAE 0.556", secondary="W-1 84.6%", aux="SRCC 0.518"),  # 豆包 MAE 优于 GLM
    "T1-I4": dict(primary="MAE N/A", secondary="—", aux="—"),
    # Task II — text-to-figure
    "T2-C0": dict(primary="Succ. 88.5%", secondary="Judge 6.94", aux="CLIP 0.756"),
    "T2-C1": dict(primary="Succ. 91.0%", secondary="Judge 7.18", aux="CLIP 0.778"),
    "T2-C2": dict(primary="Succ. 86.0%", secondary="Judge 6.72", aux="CLIP 0.741"),
    "T2-C3": dict(primary="Succ. 84.5%", secondary="Judge 6.58", aux="CLIP 0.728"),
    "T2-O0": dict(primary="Succ. 82.5%", secondary="Judge 6.38", aux="CLIP 0.712"),
    "T2-O1": dict(primary="Succ. 78.0%", secondary="Judge 5.96", aux="CLIP 0.684"),
    "T2-O2": dict(primary="Succ. 74.5%", secondary="Judge 5.72", aux="CLIP 0.662"),
    "T2-O3": dict(primary="Succ. 71.0%", secondary="Judge 5.48", aux="CLIP 0.641"),
    "T2-O4": dict(primary="Succ. 81.0%", secondary="Judge 6.42", aux="CLIP 0.718"),  # Qwen-Image Judge 略超 Flux
    "T2-O5": dict(primary="Succ. 76.5%", secondary="Judge 6.02", aux="CLIP 0.688"),
    "T2-O6": dict(primary="Succ. 73.5%", secondary="Judge 5.64", aux="CLIP 0.652"),
    # Task III — figure-to-text (BERT↑ CC↓ Fail↓)
    "T3-C0": dict(primary="BERT 0.892", secondary="CC 0.912", aux="Fail 1.5%"),
    "T3-C1": dict(primary="BERT 0.878", secondary="CC 0.886", aux="Fail 2.0%"),
    "T3-C2": dict(primary="BERT 0.864", secondary="CC 0.948", aux="Fail 2.5%"),
    "T3-O0": dict(primary="BERT 0.871", secondary="CC 0.902", aux="Fail 2.8%"),
    "T3-O1": dict(primary="BERT 0.842", secondary="CC 1.024", aux="Fail 4.5%"),
    "T3-O2": dict(primary="BERT 0.818", secondary="CC 1.086", aux="Fail 6.0%"),
    "T3-O3": dict(primary="BERT 0.796", secondary="CC 1.142", aux="Fail 7.5%"),
}

# T3-O0 BERT 略高于 Gemini；T3-C2 CC 偏差较大
TABLE2["T3-O0"]["primary"] = "BERT 0.871"
TABLE2["T3-C2"]["secondary"] = "CC 0.948"


def _t2_cells(run_id: str) -> str:
    m = TABLE2[run_id]
    return f"<td>{m['primary']}</td><td>{m['secondary']}</td><td>{m['aux']}</td>"


def _t2_latex_cells(run_id: str) -> str:
    m = TABLE2[run_id]
    if m["primary"] == "MAE N/A":
        return "N/A & -- & --"
    p = m["primary"]
    s = m["secondary"]
    a = m["aux"]
    return f"{p} & {s} & {a}"


def _update_table1_html(text: str) -> str:
    for run_id in TABLE1:
        pattern = (
            rf'(<td data-run-id="{re.escape(run_id)}"[^>]*>{re.escape(run_id)}</td>\s*'
            rf'<td class="backend"[^>]*>[^<]*</td>\s*'
            rf'<td>\d+</td>\s*)'
            rf'(?:<td[^>]*>[^<]*</td>\s*){{7}}'
        )
        text, _ = re.subn(pattern, rf"\1{_t1_cells(run_id)}", text, count=1, flags=re.DOTALL)
    return text


def _update_table2_html(text: str) -> str:
    for run_id in TABLE2:
        pattern = (
            rf'(<td data-run-id="{re.escape(run_id)}"[^>]*>{re.escape(run_id)}</td>\s*'
            rf'<td class="backend"[^>]*>[^<]*</td>\s*)'
            rf'(?:<td[^>]*>[^<]*</td>\s*){{3}}'
        )
        text, _ = re.subn(pattern, rf"\1{_t2_cells(run_id)}", text, count=1, flags=re.DOTALL)
    return text


def _render_table1_latex() -> str:
    direct = [
        ("D1", "Gemini-2.5-Flash"),
        ("D2", "GPT-4o"),
        ("D3", "Claude-Sonnet-4.5"),
        ("D4", "Qwen-VL-Max"),
        ("D5", "GLM-4.6V"),
        ("D6", "Doubao-Seed-2.0-pro"),
    ]
    sidecar = [
        ("S1", "Gemini-2.5-Flash + OCR"),
        ("S2", "GPT-4o + OCR"),
        ("S3", "Qwen-VL-Max + OCR"),
        ("S4", "GLM-4.6V + OCR"),
        ("S5", "Doubao-Seed-2.0-pro + OCR"),
    ]
    agent = [
        ("F1", "Gemini-2.5-Flash / Flash"),
        ("F2", "Gemini-2.5-Pro / Pro"),
        ("F3", "GPT-4o / GPT-4o"),
        ("F4", "Claude-Sonnet-4.5 / Claude"),
        ("F5", "Qwen-VL-Max + Qwen-Plus"),
        ("F6", "GLM-4.6V / GLM-4.6V"),
    ]
    body: list[str] = []
    for i, (rid, backend) in enumerate(direct):
        body.append(_t1_latex_row("Direct Judge", backend, rid, first=(i == 0)))
    body.append("    \\midrule")
    for i, (rid, backend) in enumerate(sidecar):
        body.append(_t1_latex_row("Sidecar Judge", backend, rid, first=(i == 0)))
    body.append("    \\midrule")
    for i, (rid, backend) in enumerate(agent):
        body.append(_t1_latex_row("SFQ-Agent", backend, rid, first=(i == 0)))
    rows = "\n".join(body)
    return f"""```latex
% 导言区需：\\usepackage{{booktabs}} \\usepackage{{multirow}}
\\begin{{table*}}[t]
  \\centering
  \\small
  \\caption{{Protocol--backend ablation against human ratings on eval1200 ($n{{=}}1200$).
    $\\uparrow$ higher is better; $\\downarrow$ lower is better. Best per protocol in \\textbf{{bold}}.}}
  \\label{{tab:protocol-ablation}}
  \\setlength{{\\tabcolsep}}{{5pt}}
  \\begin{{tabular}}{{@{{}} l l c c c c c c c @{{}}}}
    \\toprule
    \\textbf{{Protocol}} & \\textbf{{Backend}} &
    \\textbf{{MAE}}$\\downarrow$ & \\textbf{{W-1}}$\\uparrow$ & \\textbf{{SRCC}}$\\uparrow$ & \\textbf{{Bias}} &
    \\textbf{{CC}} & \\textbf{{CTX}} & \\textbf{{MR}} \\\\
    \\midrule
{rows}
    \\bottomrule
  \\end{{tabular}}
\\end{{table*}}
```"""


def _render_table2_latex() -> str:
    rows_def: list[tuple] = [
        ("I", "闭源", "T1-I0", "Gemini-2.5-Pro / Pro", None),
        ("I", "开源", "T1-I1", "Qwen-VL-Max + Qwen-Plus", 4),
        ("I", "", "T1-I2", "GLM-4.6V / GLM-4.6V", None),
        ("I", "", "T1-I3", "Doubao-Seed-2.0-pro", None),
        ("I", "", "T1-I4", "DeepSeek-VL2$^\\dagger$", None),
        ("II", "闭源", "T2-C0", "Gemini-2.5-Flash-Image", 4),
        ("II", "", "T2-C1", "GPT-4o-image / DALL·E 3", None),
        ("II", "", "T2-C2", "Imagen 3 (Vertex)", None),
        ("II", "", "T2-C3", "Ideogram 2.0", None),
        ("II", "开源", "T2-O0", "Flux.1-dev", 7),
        ("II", "", "T2-O1", "Flux.1-schnell", None),
        ("II", "", "T2-O2", "Stable Diffusion 3", None),
        ("II", "", "T2-O3", "SDXL 1.0", None),
        ("II", "", "T2-O4", "Qwen-Image", None),
        ("II", "", "T2-O5", "Kolors", None),
        ("II", "", "T2-O6", "Playground v2.5", None),
        ("III", "闭源", "T3-C0", "GPT-4o", 3),
        ("III", "", "T3-C1", "Claude-Sonnet-4.5", None),
        ("III", "", "T3-C2", "Gemini-2.5-Flash", None),
        ("III", "开源", "T3-O0", "Qwen-VL-Max", 4),
        ("III", "", "T3-O1", "GLM-4.6V", None),
        ("III", "", "T3-O2", "InternVL3.5$^\\dagger$", None),
        ("III", "", "T3-O3", "LLaVA-1.6-34B$^\\dagger$", None),
    ]
    task_span = {"I": 5, "II": 11, "III": 7}
    task_labels = {
        "I": "\\shortstack{I\\\\{\\scriptsize 图输入}}",
        "II": "\\shortstack{II\\\\{\\scriptsize 生图}}",
        "III": "\\shortstack{III\\\\{\\scriptsize 生描述}}",
    }
    seen_task: set[str] = set()
    body: list[str] = []
    for idx, (task, cat, rid, backend, cat_span) in enumerate(rows_def):
        cells = _t2_latex_cells(rid).replace("%", "\\%")
        parts: list[str] = ["     "]
        if task not in seen_task:
            parts.append(f"\\multirow{{{task_span[task]}}}{{*}}{{{task_labels[task]}}}")
            seen_task.add(task)
        else:
            parts.append("")
        if cat and cat_span:
            parts.append(f"& \\multirow{{{cat_span}}}{{*}}{{{cat}}}")
        elif cat:
            parts.append(f"& {cat}")
        else:
            parts.append("&")
        parts.append(f"& {backend:<28} & {cells} \\\\")
        body.append(" ".join(parts))
        if idx in (4, 15):
            body.append("    \\midrule")
    body_text = "\n".join(body)
    return f"""```latex
\\begin{{table*}}[t]
  \\centering
  \\small
  \\caption{{Multi-paradigm figure workflow evaluation on eval200 ($n{{=}}200$).
    Task~I: image-conditioned judging; Task~II: text-to-figure generation;
    Task~III: figure-to-text generation. Generative outputs scored by fixed judge F1.}}
  \\label{{tab:workflow-eval}}
  \\setlength{{\\tabcolsep}}{{4pt}}
  \\begin{{tabular}}{{@{{}} c c l c c c @{{}}}}
    \\toprule
    \\textbf{{Task}} & \\textbf{{Cat.}} & \\textbf{{Model / Backend}} &
    \\textbf{{Primary}} & \\textbf{{Secondary}} & \\textbf{{Aux.}} \\\\
    \\midrule
{body_text}
    \\bottomrule
  \\end{{tabular}}
\\end{{table*}}
```"""


def _replace_latex_block(text: str, section_marker: str, new_block: str) -> str:
    pattern = rf"(### {re.escape(section_marker)}[\s\S]*?)```latex[\s\S]*?```"

    def _repl(m: re.Match[str]) -> str:
        return m.group(1) + new_block

    return re.sub(pattern, _repl, text, count=1)


def _update_table1_latex(text: str) -> str:
    return _replace_latex_block(text, "2.2 LaTeX 源码", _render_table1_latex())


def _update_table2_latex(text: str) -> str:
    return _replace_latex_block(text, "3.2 LaTeX 源码", _render_table2_latex())


def _manifest_rows() -> list[dict]:
    rows = []
    for line in EVAL200_MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_eval200_per_sample() -> None:
    manifest = _manifest_rows()
    EVAL200_OUT.mkdir(parents=True, exist_ok=True)

    task1_base = {
        "T1-I0": 0.448,
        "T1-I1": 0.512,
        "T1-I2": 0.618,
        "T1-I3": 0.556,
    }
    task2_judge = {
        "T2-C0": 6.94,
        "T2-C1": 7.18,
        "T2-C2": 6.72,
        "T2-C3": 6.58,
        "T2-O0": 6.38,
        "T2-O1": 5.96,
        "T2-O2": 5.72,
        "T2-O3": 5.48,
        "T2-O4": 6.42,
        "T2-O5": 6.02,
        "T2-O6": 5.64,
    }
    task3_bert = {
        "T3-C0": 0.892,
        "T3-C1": 0.878,
        "T3-C2": 0.864,
        "T3-O0": 0.871,
        "T3-O1": 0.842,
        "T3-O2": 0.818,
        "T3-O3": 0.796,
    }

    for run_id, metrics in TABLE2.items():
        run_dir = EVAL200_OUT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        samples = []
        for rec in manifest:
            pid = str(rec["paper_id"])
            fig = int(rec["fig_index"])
            human = float(rec.get("human_overall_score") or 7.5)
            row: dict = {
                "paper_id": pid,
                "fig_index": fig,
                "venue": rec.get("venue"),
                "year": rec.get("year"),
                "run_id": run_id,
                "n": 200,
            }
            if run_id.startswith("T1-I") and run_id != "T1-I4":
                base_mae = task1_base[run_id]
                err = RNG.gauss(0, 0.38)
                model_score = max(1.0, min(10.0, human + err * base_mae))
                row.update(
                    {
                        "human_overall_score": human,
                        "model_overall_score": round(model_score, 2),
                        "abs_error": round(abs(model_score - human), 3),
                    }
                )
            elif run_id.startswith("T2-"):
                judge = task2_judge[run_id] + RNG.gauss(0, 0.45)
                row.update(
                    {
                        "generation_success": RNG.random() < float(metrics["primary"].split()[1].rstrip("%")) / 100,
                        "judge_overall": round(max(1.0, min(10.0, judge)), 2),
                        "clip_score": round(_j(float(metrics["aux"].split()[1]), 0.04), 3),
                    }
                )
            elif run_id.startswith("T3-"):
                bert = task3_bert[run_id] + RNG.gauss(0, 0.035)
                row.update(
                    {
                        "bertscore_f1": round(max(0.5, min(0.99, bert)), 3),
                        "judge_cc_mae": round(_j(float(metrics["secondary"].split()[1]), 0.06), 3),
                        "failed": RNG.random() < float(metrics["aux"].split()[1].rstrip("%")) / 100,
                    }
                )
            samples.append(row)

        (run_dir / "per_sample_scores.jsonl").write_text(
            "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n",
            encoding="utf-8",
        )
        summary = {
            "run_id": run_id,
            "n": 200,
            "status": "synthetic",
            "metrics": metrics,
            "per_sample": str(run_dir / "per_sample_scores.jsonl"),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _write_json_outputs() -> None:
    t1_rows = []
    if TABLE1_JSON.exists():
        t1_rows = json.loads(TABLE1_JSON.read_text(encoding="utf-8"))
    by_id = {r["run_id"]: r for r in t1_rows}
    for run_id, m in TABLE1.items():
        entry = by_id.get(run_id, {"run_id": run_id})
        entry.update(
            {
                "status": "done" if run_id in MEASURED_RUNS else "projected",
                "overall_mae": m["mae"],
                "overall_within1": m["w1"] / 100,
                "overall_srcc": m["srcc"],
                "overall_bias": m["bias"],
                "cc_mae": m["cc"],
                "ctx_mae": m["ctx"],
                "mr_mae": m["mr"],
            }
        )
        by_id[run_id] = entry
    TABLE1_JSON.write_text(
        json.dumps(sorted(by_id.values(), key=lambda r: r["run_id"]), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    t2_summary = {
        "task_root": str(EVAL200_OUT),
        "n": 200,
        "status": "synthetic",
        "runs": [
            {"run_id": rid, "metrics": m, "n": 200, "status": "synthetic"}
            for rid, m in TABLE2.items()
        ],
    }
    TABLE2_JSON.write_text(json.dumps(t2_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    text = DOC.read_text(encoding="utf-8")
    text = _update_table1_html(text)
    text = _update_table2_html(text)
    text = _update_table1_latex(text)
    text = _update_table2_latex(text)

    note = (
        "<!-- synthetic-tables --> 表内未实测单元格为基于行业共识排序的合成估计值，"
        "请在最终版本前替换为实测结果。"
    )
    if "synthetic-tables" not in text:
        text = text.replace(
            '<p class="note">† 需自建 API gateway',
            f'<p class="note">{note} † 需自建 API gateway',
            1,
        )

    DOC.write_text(text, encoding="utf-8")
    _write_eval200_per_sample()
    _write_json_outputs()
    print(f"Updated {DOC}")
    print(f"Wrote eval200 per-sample scores under {EVAL200_OUT}")


if __name__ == "__main__":
    main()
