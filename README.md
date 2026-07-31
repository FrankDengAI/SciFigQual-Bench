# SciFigQual-Bench

**A Benchmark for Scientific Figure Quality Assessment with Full-Manuscript Context**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


<p align="center">
  <img src="docs/figures/fig3_sfq_agent.png" alt="SFQ-Agent scoring pipeline" width="92%">
</p>
<p align="center"><em>SFQ-Agent: L1 gating, parallel vision/language evidence, cross-modal judge, and deterministic Runner aggregation.</em></p>

---
The data is in: https://huggingface.co/datasets/haihanlamu/SciFigQual-Bench
## Table of Contents

- [Key Highlights](#key-highlights)
- [Overview](#overview)
- [Benchmark at a Glance](#benchmark-at-a-glance)
- [Construction Pipeline](#construction-pipeline)
- [Dataset Statistics](#dataset-statistics)
- [SFQ-Agent](#sfq-agent)
- [Five-Dimensional Rubric](#five-dimensional-rubric)
- [Judge Protocols](#judge-protocols)
- [eval1200 Experiment Matrix](#eval1200-experiment-matrix)
- [Quick Start](#quick-start)
- [Repository Layout](#repository-layout)
- [Related Releases](#related-releases)
- [License](#license)

---

## Key Highlights

- **Full-manuscript context** — Each instance is \((I, c, \mathcal{T}, m)\): figure image, caption, citing paragraphs from the source PDF, and metadata.
- **Large-scale gold standard** — **7,609** curated figures from **1,144** qualified CS papers; **6,308** expert-rated instances with aggregated five-dimensional scores.
- **Top venues, 2020–2025** — ACL, EMNLP, ICML, NeurIPS; ~355k index-driven citing-paragraph records.
- **Five-dimensional rubric** — Visual Clarity (VC), Structure & Layout (SL), Caption Consistency (CC), Context Consistency (CTX), Misleading Risk (MR) on a unified 1–10 scale with L1 evidence gating.
- **SFQ-Agent** — Staged evidence collection + cross-modal fusion; on **eval1200** (*n*=1,200), SFQ-Agent with GPT-5.6-Sol (F3) reaches **MAE 0.418** and **93.4%** within-1 agreement vs. human gold.
- **29 protocol–backend configurations** — Direct (D1–D11), Sidecar + OCR (S1–S9), SFQ-Agent (F1–F9) ablations in [`configs/eval1200_ablation.yaml`](configs/eval1200_ablation.yaml).

---

## Overview

Scientific figure quality in peer review is inherently **tri-modal**: reviewers cross-check what is visible in the figure \(I\), what the caption \(c\) claims, and what citing paragraphs \(\mathcal{T}\) assert. Natural IQA, AIGC alignment, chart QA, and figure reasoning benchmarks typically score **isolated** visuals without manuscript evidence.

SciFigQual-Bench closes this gap for **published** CS conference figures:

1. **Acquire & curate** PDFs from four top venues (62,694 raw → 7,609 clean figures).
2. **Bind context** via PDF index patterns (not abstract-only heuristics).
3. **Annotate** with a calibrated five-dimensional rubric and release a fixed public test split **eval1200**.

This repository ships the **evaluation code** (three judge protocols + benchmark scripts). The full dataset and pre-computed predictions are distributed as companion packages (see [Related Releases](#related-releases)).

---

## Benchmark at a Glance

| Item | Count / value |
|------|----------------|
| Raw PDF corpus | 62,694 |
| Qualified papers | 1,144 |
| Curated figures | 7,609 |
| Human gold instances | 6,308 |
| Public test split `eval1200` | 1,200 (stratified) |
| Venues | ACL, EMNLP, ICML, NeurIPS |
| Years | 2020–2025 |
| Mean overall human score | 8.05 (on rated subset) |

---

## Construction Pipeline

<p align="center">
  <img src="docs/figures/fig2_construction_pipeline.png" alt="SciFigQual-Bench construction pipeline" width="92%">
</p>
<p align="center"><em>Corpus collection, structure-aware extraction, context binding, five-dimensional rubric, and expert validation.</em></p>

The pipeline has five deterministic modules (given fixed PDFs and preprocessing seeds):

| Module | Description |
|--------|-------------|
| **1 — Corpus acquisition** | Crawl OpenReview, ACL Anthology, PMLR; normalize PDFs; SHA-256 dedup; venue/year indexing. |
| **2 — Figure extraction** | Marker layout parsing @ 300 DPI; caption–figure matching; five-stage curation (S1–S5). |
| **3 — Context binding** | PyMuPDF index patterns → citing paragraphs per figure; ~355k paragraph records. |
| **4 — Human annotation** | Pilot calibration; dual-rater holdout; L1 gating hides CC/CTX when evidence is absent. |
| **5 — Release packaging** | JSONL + images; stratified **eval1200** split; unified judge I/O schema for fair protocol comparison. |

---

## Dataset Statistics

<p align="center">
  <img src="docs/figures/fig4_dataset_statistics.png" alt="Dataset statistics panels" width="92%">
</p>
<p align="center"><em>Venue–topic radar, construction funnel, score distribution, domain counts, temporal coverage, and per-dimension means.</em></p>

Key takeaways from the rated subset (*n*=6,308):

- **Funnel yield** \(\eta = N/K_0 \approx 12.1\%\) reflects strict curation, not noisy crawling.
- **CC is the weakest axis** (mean 7.42) vs. SL (8.58)—caption–figure mismatches dominate real defects, motivating manuscript-grounded CC and CTX evaluation.
- **Temporal coverage** peaks in 2024–2025, supporting evaluation on modern plotting styles.

---

## SFQ-Agent

Caption and context consistency are hard for monolithic VLMs. **SFQ-Agent** (Scientific Figure Quality Agent) implements evidence-grounded judging in four stages:

| Stage | Role |
|-------|------|
| **0 — Input & gating** | Load \((I, c, \mathcal{T})\); apply L1 dimension hiding. |
| **1 — Vision evidence** | PaddleOCR-VL + CV descriptors → structured `visual_facts` for VC/SL. |
| **2 — Language evidence** | LLM reads caption + citing text only (no pixels) for CC/CTX cues. |
| **3 — Cross-modal judge** | Fuse tracks; score CC, CTX, MR; detect visual–text conflicts. |
| **4 — Runner** | Deterministic aggregation; rule-based MR caps; traceable score lineage. |

Direct and Sidecar judges are ablations on the same rubric and I/O schema—performance gaps reflect **judging strategy**, not data representation differences.

---

## Five-Dimensional Rubric

Scores are on a **1–10** Likert scale per dimension. Overall score is a gated mean over available dimensions.

| Dim | Name | What it measures |
|-----|------|------------------|
| **VC** | Visual Clarity | Readability, resolution, contrast, label legibility |
| **SL** | Structure & Layout | Panel organization, alignment, visual hierarchy |
| **CC** | Caption Consistency | Caption faithfully describes visible content |
| **CTX** | Context Consistency | Figure supports claims in citing paragraphs |
| **MR** | Misleading Risk | Truncated axes, missing baselines, deceptive encodings |

**L1 gating:** CC is null when caption is absent; CTX is null when citing text is absent; instances lacking both are excluded from scoring.

Prompt templates: [`model_scoring/prompts/`](model_scoring/prompts/) (Direct/Sidecar) and [`model_scoring/agent_scoring/prompts/`](model_scoring/agent_scoring/prompts/) (Agent).

---

## Judge Protocols

| Protocol | API calls / figure | Description |
|----------|-------------------|-------------|
| **Direct Judge** | 1 | Single VLM prompt over \((I, c, \mathcal{T})\) |
| **Sidecar Judge** | 1 | Direct prompt + PaddleOCR-VL / CV side features |
| **SFQ-Agent** | 3 | Staged VLM evidence → LLM evidence → cross-modal judge |

Implementation entry points:

- Direct / Sidecar: [`model_scoring/score_and_upload.py`](model_scoring/score_and_upload.py)
- SFQ-Agent: [`model_scoring/agent_scoring/run_agent_score.py`](model_scoring/agent_scoring/run_agent_score.py)

---

## eval1200 Experiment Matrix

Fixed public test split: **1,200** figures stratified by venue and figure type ([`configs/human_eval_subset_1200.jsonl`](configs/human_eval_subset_1200.jsonl)).

| Family | Run IDs | Backends (examples) |
|--------|---------|---------------------|
| Direct | D1–D11 | Gemini, GPT, Claude, Qwen, GLM, Doubao, Llama, Pixtral, Nova, Opus, InternVL |
| Sidecar | S1–S9 | Same subset + OCR side features |
| SFQ-Agent | F1–F9 | Staged agent with matched VLM/LLM pairs |

Full matrix: [`configs/eval1200_ablation.yaml`](configs/eval1200_ablation.yaml) (29 runs).

Metrics vs. human gold: [`scripts/benchmark/evaluate_vs_human.py`](scripts/benchmark/evaluate_vs_human.py)  
Aggregate tables: [`scripts/benchmark/aggregate_experiment_results.py`](scripts/benchmark/aggregate_experiment_results.py)

---

## Quick Start

### Install

```bash
git clone <your-repo-url>
cd SciFigQual-Bench
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Optional GPU extras (PaddleOCR-VL for Sidecar): see [`pyproject.toml`](pyproject.toml) `[dependency-groups.gpu]`.

### API keys (live inference)

**Default:** vendor HTTP calls are **stubbed** via [`model_scoring/live_api_guard.py`](model_scoring/live_api_guard.py) so the public repo ships without secrets:

```
NotImplementedError: Set API keys in .env to run live inference (see README).
```

To run live scoring:

1. Copy [`.env.example`](.env.example) → `.env` and set provider keys.
2. Remove or bypass `block_live_inference()` in client entry points, or restore upstream implementations from your private fork.

### Dataset layout (local)

Download the HuggingFace release and point:

```
datasets/full/
├── figures.jsonl      # 6,308 instances
├── human_means.csv
└── images/            # 6,308 PNG crops
```

Use `configs/human_eval_subset_1200.jsonl` for the eval1200 manifest.

### Run the experiment matrix (dry-run)

```bash
python scripts/benchmark/run_experiment_matrix.py --config configs/eval1200_ablation.yaml
```

### Score one configuration (after enabling live APIs)

```bash
python model_scoring/score_and_upload.py \
  --batch-mode paper \
  --manifest configs/human_eval_subset_1200.jsonl \
  --skip-upload \
  --provider gemini \
  --model gemini-2.5-flash \
  --prompt-mode baseline
```

### Evaluate predictions vs. human gold

```bash
python scripts/benchmark/evaluate_vs_human.py \
  --predictions outputs/benchmark_eval1200/D1/results.jsonl \
  --human datasets/eval1200/human_means.csv
```

Pre-computed predictions for all 29 runs ship in the companion **supporting materials** package.

### Regenerate README figures from paper PDFs

```bash
python scripts/export_readme_figures.py --paper-dir /path/to/paper
```

---

## Repository Layout

```
.
├── model_scoring/           # Direct & Sidecar judges, prompts, provider registry
│   └── agent_scoring/       # SFQ-Agent (3-stage pipeline)
├── src/cs64/                # Data I/O, OCR/CV feature pipeline (Sidecar)
├── scripts/
│   ├── benchmark/           # eval1200 matrix, metrics, synthetic prediction tools
│   ├── build_datasets.py
│   └── export_readme_figures.py
├── configs/
│   ├── eval1200_ablation.yaml
│   └── human_eval_subset_1200.jsonl
└── docs/figures/            # README figures (Fig 2–4 from paper)
```

---

## Related Releases

This folder is **package 1 of 3** in the SciFigQual upload bundle:

| Package | Contents |
|---------|----------|
| **1 — GitHub code** (this repo) | Judges, rubric, benchmark scripts |
| **2 — HuggingFace benchmark** | 6,308 gold instances + PNG + eval1200 split |
| **3 — Supporting eval results** | 29×1,200 predictions + `table1_metrics.json` |

Companion paths in the monorepo upload layout:

- `../2_huggingface_benchmark/`
- `../3_supporting_eval_results/`

---

## License

Released under the **MIT License**. See companion dataset card for data terms.
