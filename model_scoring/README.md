# `model_scoring/` — Evaluate (Stage 5)

Pipeline **Stage 5 (Evaluate)**: score each figure on five dimensions
(`visual_clarity`, `structure_layout`, `caption_consistency`,
`context_consistency`, `misleading_risk`) + `overall_score`, written to
`results.parquet`. Three workflows share one rubric/schema:
**Direct Judge** and **Sidecar Judge** (`score_and_upload.py`) and the **SFQ-Agent
agent** (`agent_scoring/run_agent_score.py`); an orthogonal paper-batch mode adds
paper-level context. Gemini 2.5-flash in production, Claude as a cross-check.

📖 **Full documentation:**
[`wiki/02-scoring/00-overview.md`](../wiki/02-scoring/00-overview.md) (overview),
[`wiki/02-scoring/02-model-scoring.md`](../wiki/02-scoring/02-model-scoring.md) (baseline/with-features/paper-batch),
[`wiki/02-scoring/03-agent-scoring.md`](../wiki/02-scoring/03-agent-scoring.md) (agent),
[`wiki/02-scoring/01-dimensions-and-rubric.md`](../wiki/02-scoring/01-dimensions-and-rubric.md) (rubric).
Run commands: [`wiki/appendix/02-reproduce.md`](../wiki/appendix/02-reproduce.md).

> See [`wiki/index.md`](../wiki/index.md) for the whole project.
