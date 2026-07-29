# Model Scoring Prompt Structure

This note records the consolidated prompt layout after version cleanup.

## Final Active Prompt Folders

Only these prompt folders are active:

- `prompts/`
- `agent_scoring/prompts/`

Older prompt packages are archived under `oldversion/`.

## Non-Agent Gemini Prompts

Loaded by `model_scoring/providers.py`, `model_scoring/runner.py`, and
`model_scoring/paper_runner.py`.

Single-figure and paper-batch scoring both use the same v3 paper prompt
standards. In single-figure mode the runner sends one target figure at a time
through the paper prompt path.

- `prompts/baseline_paper_batch.md`
- `prompts/with_features_paper_batch.md`

The old single-figure prompts, old v2 paper-batch files, and previous `_v3`
named copies are archived in `oldversion/prompts/`.

## Agent Prompts

Loaded by `model_scoring/agent_scoring/runner.py`.

Single-figure and paper-batch agent scoring both use the same v5 paper-batch
evidence interaction. In single-figure mode the runner builds the same
`paper_batch_json` payload with one target figure, so VLM evidence, LLM evidence,
and final judge schemas match paper-batch mode.

- `agent_scoring/prompts/vlm_evidence_report_paper_batch.md`
- `agent_scoring/prompts/llm_evidence_report_paper_batch.md`
- `agent_scoring/prompts/final_judge_paper_batch.md`

The old single-figure agent prompts are archived in
`oldversion/agent_scoring/figure_v2/`.

## Archive

Historical prompt packages are kept under:

- `oldversion/prompts/v2_paper_batch/`
- `oldversion/prompts/figure_v2/`
- `oldversion/prompts/v3_named_copies/`
- `oldversion/agent_scoring/figure_v2/`
- `oldversion/v3_prompt/`
- `oldversion/v4_prompt/`
- `oldversion/v5_prompt/`

These files are retained for comparison and audit history, not as the primary
runtime prompt entry points.
