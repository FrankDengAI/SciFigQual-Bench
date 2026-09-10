# SFQ-Agent (`agent_scoring/`)

Auditable staged scoring:

1. **Vision evidence** — VLM reads the figure (+ optional features) → structured visual facts (VC / SL focus).
2. **Language evidence** — LLM reads caption + citing paragraphs only → text facts (CC / CTX focus).
3. **Cross-modal judge** — fuse tracks; score remaining dimensions; flag visual–text conflicts.
4. **Runner aggregation** — deterministic score assembly with rule-based MR caps.

Entry point: `run_agent_score.py`  
Active prompts: `prompts/vlm_evidence_report_paper_batch.md`, `llm_evidence_report_paper_batch.md`, `final_judge_paper_batch.md`.
