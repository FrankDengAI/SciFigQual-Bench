# Agent Scoring (SFQ-Agent)

The auditable scoring workflow: a VLM stage (image + features → visual facts &
scores), an LLM stage (caption + context → text facts), and a final judge that
combines structured evidence — so every score is traceable. Final rows keep the
shared contract (`source_type=model`, `source_name=<model_label>`). Entry:
`run_agent_score.py`. Active prompts are v5 (`prompts/`).

📖 **Full documentation:** [`wiki/02-scoring/03-agent-scoring.md`](../../wiki/02-scoring/03-agent-scoring.md)
— scoring authority, the four design stages, code-enforced constraints, and the
misleading-risk decomposition.

> See [`wiki/index.md`](../../wiki/index.md) for the whole project.
