from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import polars as pl
import typer
from pydantic import BaseModel

AGENT_DIR = Path(__file__).resolve().parent
MODEL_SCORING_DIR = AGENT_DIR.parent
for path in (str(AGENT_DIR), str(MODEL_SCORING_DIR)):
    if path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(MODEL_SCORING_DIR))
sys.path.insert(0, str(AGENT_DIR))

from feature_payload import (  # noqa: E402
    feature_lookup_map,
    load_feature_table,
)
from io_utils import (  # noqa: E402
    REPO_ID,
    append_result_jsonl,
    filter_by_manifest,
    limit_to_first_papers,
    load_context_from_hf,
    load_env,
    load_existing_results,
    load_features_from_hf,
    load_figures,
    merge_results,
    prepare_rows,
    upload_results,
    write_audit,
    write_error_audit,
)
from paper_inputs import (  # noqa: E402
    build_context_payload,
    context_key,
    context_lookup_map,
    filter_rows_with_text_evidence,
    format_paper_payload,
    load_context_table,
    paper_prompt_payload,
)
from providers import generate_text, generate_vision_batch  # noqa: E402
from provider_registry import agent_llm_model, validate_provider  # noqa: E402
from schema import (  # noqa: E402
    DIMENSIONS,
    PaperAgentFinalAssessment,
    V5PaperLlmEvidenceReport,
    V5PaperVlmEvidenceReport,
    available_dimensions,
    compute_overall,
    validate_paper_final,
)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
PROMPT_DIR = AGENT_DIR / "prompts"


def load_prompt(name: str) -> str:
    path = PROMPT_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def chunks(values: list[dict], size: int) -> list[list[dict]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def no_text_evidence_message(row: dict) -> str:
    return (
        f"Skipped {row['paper_id']} figure {row['fig_index']} because it has neither "
        "caption nor usable context."
    )


def model_json(value: BaseModel | dict | list | None) -> str:
    if isinstance(value, BaseModel):
        payload = value.model_dump()
    else:
        payload = value
    return json.dumps(payload, ensure_ascii=False, indent=2)


def without_feature_json(value):
    if isinstance(value, dict):
        return {
            key: without_feature_json(child)
            for key, child in value.items()
            if key != "feature_json"
        }
    if isinstance(value, list):
        return [without_feature_json(item) for item in value]
    return value


def vlm_only_payload(payload: dict) -> dict:
    return {
        "target_figures": [
            {
                "fig_index": target["fig_index"],
                "available_dimensions": target.get("available_dimensions", []),
                "feature_json": target.get("feature_json"),
            }
            for target in payload.get("target_figures", [])
        ]
    }


def retry_feedback(error: Exception | None) -> str:
    if error is None:
        return ""
    return (
        "\n\nPrevious attempt failed validation or parsing with this error:\n"
        f"{error}\n"
        "Correct only the invalid part. Return exactly the requested target figures "
        "and all required available dimensions."
    )


def call_with_retry_feedback(fn, max_retries: int) -> ResponseModel:
    if max_retries < 1:
        raise typer.BadParameter("--max-retries must be >= 1.")
    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            return fn(retry_feedback(last_error))
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def validate_report_figures(report_name: str, figures: list[BaseModel], expected: set[int]) -> None:
    seen = {int(item.fig_index) for item in figures}
    missing = sorted(expected - seen)
    extra = sorted(seen - expected)
    if missing:
        raise ValueError(f"{report_name} missing target figures: {missing}")
    if extra:
        raise ValueError(f"{report_name} included non-target figures: {extra}")


def apply_v5_visual_scores_and_caps(
    final: PaperAgentFinalAssessment,
    vlm_evidence: V5PaperVlmEvidenceReport,
    llm_evidence: V5PaperLlmEvidenceReport,
) -> PaperAgentFinalAssessment:
    vlm_by_fig = {int(item.fig_index): item for item in vlm_evidence.figures}
    llm_by_fig = {int(item.fig_index): item for item in llm_evidence.figures}
    for item in final.figures:
        fig_index = int(item.fig_index)
        vlm_item = vlm_by_fig.get(fig_index)
        if vlm_item is not None:
            item.visual_clarity = vlm_item.visual_scores.visual_clarity
            item.structure_layout = vlm_item.visual_scores.structure_layout
        llm_item = llm_by_fig.get(fig_index)
        if llm_item is None:
            continue
        flags = llm_item.rule_flags
        if item.caption_consistency is not None and flags.caption_score_cap is not None:
            item.caption_consistency.score = min(
                int(item.caption_consistency.score),
                int(flags.caption_score_cap),
            )
        if item.context_consistency is not None and flags.context_score_cap is not None:
            item.context_consistency.score = min(
                int(item.context_consistency.score),
                int(flags.context_score_cap),
            )
        if item.misleading_risk is not None and vlm_item is not None:
            fused = fuse_misleading_risk_score(vlm_item, llm_item)
            if flags.misleading_score_cap is not None:
                fused = min(fused, int(flags.misleading_score_cap))
            item.misleading_risk.score = fused
    return final


def _parse_risk_band(band: str) -> tuple[int, int]:
    cleaned = band.strip().replace("–", "-")
    if "-" in cleaned:
        lo, hi = cleaned.split("-", 1)
        return int(lo.strip()), int(hi.strip())
    value = int(float(cleaned))
    return value, value


def fuse_misleading_risk_score(vlm_item, llm_item) -> int:
    """Deterministic MR fusion from v5 visual/text evidence (Runner hard constraint)."""
    lo, hi = _parse_risk_band(vlm_item.visual_misleading_risk.intrinsic_risk_band)
    visual_mid = (lo + hi) / 2.0
    severity = llm_item.text_misleading_adjustment.severity.lower().strip()

    if severity == "none":
        score = visual_mid
    elif severity == "minor":
        score = max(lo, visual_mid - 1.0)
        if lo >= 9:
            score = max(score, 8.0)
        elif lo >= 7:
            score = max(score, 7.0)
    elif severity == "moderate":
        score = visual_mid if hi <= 5 else 6.5
    elif severity == "severe":
        score = visual_mid if hi <= 3 else 4.5
    elif severity == "contradiction":
        score = 2.0
    else:
        score = visual_mid

    return max(1, min(10, round(score)))


def scored_row_from_final(row: dict, label: str, final) -> dict:
    dim_scores = {
        dim: (getattr(final, dim).score if getattr(final, dim) is not None else None)
        for dim in DIMENSIONS
    }
    return {
        "paper_id": row["paper_id"],
        "venue": row["venue"],
        "year": row["year"],
        "fig_index": row["fig_index"],
        "source_type": "model",
        "source_name": label,
        **dim_scores,
        "overall_score": compute_overall(dim_scores),
        "summary": final.summary,
        "suggestion": final.suggestion,
        "visual_clarity_reason": final.visual_clarity.reason,
        "structure_layout_reason": final.structure_layout.reason,
        "caption_consistency_reason": (
            final.caption_consistency.reason if final.caption_consistency is not None else None
        ),
        "context_consistency_reason": (
            final.context_consistency.reason if final.context_consistency is not None else None
        ),
        "misleading_risk_reason": final.misleading_risk.reason,
        "annotated_at": datetime.now(UTC).isoformat(),
    }


def feature_status_for_row(row: dict, feature_lookup: dict[tuple[str, int], dict]) -> str:
    return (
        "matched" if (str(row["paper_id"]), int(row["fig_index"])) in feature_lookup else "missing"
    )


def _append_scored_jsonl(output_jsonl: str | None, rows: list[dict]) -> None:
    if not output_jsonl:
        return
    for row in rows:
        append_result_jsonl(output_jsonl, row)


def echo_feature_coverage(
    figures: pl.DataFrame, feature_lookup: dict[tuple[str, int], dict]
) -> None:
    total = len(figures)
    if total == 0:
        return
    matched = sum(
        1
        for row in figures.iter_rows(named=True)
        if (str(row["paper_id"]), int(row["fig_index"])) in feature_lookup
    )
    missing = total - matched
    typer.echo(
        f"Feature coverage: {matched}/{total} figures matched; "
        f"{missing} will use feature_json=null."
    )


def require_feature_coverage(
    figures: pl.DataFrame,
    feature_lookup: dict[tuple[str, int], dict],
    *,
    required: bool,
) -> None:
    if not required:
        return
    missing = [
        (str(row["paper_id"]), int(row["fig_index"]))
        for row in figures.iter_rows(named=True)
        if (str(row["paper_id"]), int(row["fig_index"])) not in feature_lookup
    ]
    if missing:
        preview = ", ".join(f"{paper_id} fig {fig_index}" for paper_id, fig_index in missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... (+{len(missing) - 8} more)"
        raise typer.BadParameter(
            "Current agent paper-batch scoring requires feature_json for every target figure. "
            f"Missing {len(missing)} feature rows: {preview}{suffix}. "
            "Pass --features-parquet or ensure HF features are available."
        )


def run_agent_for_paper_chunk(
    *,
    paper_rows: list[dict],
    target_rows: list[dict],
    feature_lookup: dict[tuple[str, int], dict],
    context_lookup: dict[tuple[str, int], list[dict]],
    available_dims_by_fig: dict[int, list[str]],
    prior_assessments: list[dict],
    vlm_provider: str,
    vlm_model: str,
    llm_provider: str,
    llm_model: str,
    max_retries: int,
    include_paper_batch_context: bool,
) -> tuple[PaperAgentFinalAssessment, dict]:
    """SFQ-Agent canonical stack for one paper chunk (maps to paper Fig. L1/L3).

    Pipeline stages:
      L1 — ``available_dims_by_fig`` encodes hide-not-penalise (CC/CTX null, skip if no text).
      L3.1 Vision Evidence Module — VLM on image I + OCR/CV ``feature_json``.
      L3.2 Language Evidence Module — LLM on caption c + citing paragraphs T (no image).
      L3.3 MERGE — both JSON reports embedded in cross-modal judge prompt.
      L3.4 Cross-modal Judge — LLM scores CC, CTX, MR with reasons.
      L3.5 Runner — ``apply_v5_visual_scores_and_caps`` copies VC/SL, caps CC/CTX, fuses MR.
    """
    target_indices = {int(row["fig_index"]) for row in target_rows}
    payload = paper_prompt_payload(
        paper_rows=paper_rows,
        target_rows=target_rows,
        feature_lookup=feature_lookup,
        context_lookup=context_lookup,
        prior_assessments=prior_assessments,
        include_paper_batch_context=include_paper_batch_context,
    )
    for target in payload["target_figures"]:
        target["available_dimensions"] = available_dims_by_fig[int(target["fig_index"])]
    vlm_payload_text = f"vision_batch_json: {format_paper_payload(vlm_only_payload(payload))}"
    text_payload = without_feature_json(payload)
    user_payload_text = f"paper_batch_json: {format_paper_payload(text_payload)}"

    def build_vlm_evidence(feedback: str) -> V5PaperVlmEvidenceReport:
        evidence = generate_vision_batch(
            provider=vlm_provider,
            rows=target_rows,
            model=vlm_model,
            system_prompt=load_prompt("vlm_evidence_report_paper_batch"),
            user_payload_text=vlm_payload_text + feedback,
            response_model=V5PaperVlmEvidenceReport,
        )
        validate_report_figures("vlm_evidence", evidence.figures, target_indices)
        return evidence

    vlm_evidence = call_with_retry_feedback(build_vlm_evidence, max_retries)

    def build_llm_evidence(feedback: str) -> V5PaperLlmEvidenceReport:
        evidence = generate_text(
            provider=llm_provider,
            model=llm_model,
            system_prompt=load_prompt("llm_evidence_report_paper_batch"),
            user_prompt=(
                f"{user_payload_text}\n"
                "Produce compact text/feature facts, text-implied image expectations, "
                "and rule flags for every target figure. "
                "Do not score."
                f"{feedback}"
            ),
            response_model=V5PaperLlmEvidenceReport,
        )
        validate_report_figures("llm_evidence", evidence.figures, target_indices)
        return evidence

    llm_evidence = call_with_retry_feedback(build_llm_evidence, max_retries)

    def build_final(feedback: str) -> PaperAgentFinalAssessment:
        assessment = generate_text(
            provider=llm_provider,
            model=llm_model,
            system_prompt=load_prompt("final_judge_paper_batch"),
            user_prompt=(
                f"{user_payload_text}\n\n"
                f"VLM evidence report:\n{model_json(vlm_evidence)}\n\n"
                f"LLM evidence report:\n{model_json(llm_evidence)}\n"
                f"{feedback}"
            ),
            response_model=PaperAgentFinalAssessment,
        )
        assessment = apply_v5_visual_scores_and_caps(assessment, vlm_evidence, llm_evidence)
        validate_paper_final(assessment, target_indices, available_dims_by_fig)
        return assessment

    final = call_with_retry_feedback(build_final, max_retries)

    return final, {
        "paper_batch_payload": payload,
        "vlm_evidence": vlm_evidence.model_dump(),
        "llm_evidence": llm_evidence.model_dump(),
        "final": final.model_dump(),
    }


def run_agent_for_paper(
    *,
    paper_index: int,
    total_papers: int,
    paper_df: pl.DataFrame,
    feature_lookup: dict[tuple[str, int], dict],
    context_lookup: dict[tuple[str, int], list[dict]],
    label: str,
    max_figures_per_call: int,
    vlm_provider: str,
    vlm_model: str,
    llm_provider: str,
    llm_model: str,
    max_retries: int,
    include_paper_batch_context: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    scored_rows: list[dict] = []
    audit_records: list[dict] = []
    error_records: list[dict] = []

    paper_rows = list(paper_df.sort("fig_index").iter_rows(named=True))
    prior_assessments: list[dict] = []
    for chunk_index, target_rows in enumerate(
        chunks(paper_rows, max_figures_per_call),
        start=1,
    ):
        available_dims_by_fig: dict[int, list[str]] = {}
        scorable_rows: list[dict] = []
        for row in target_rows:
            context_payload = build_context_payload(context_lookup.get(context_key(row), []))
            has_caption = bool(str(row.get("caption") or "").strip())
            has_context = int(context_payload["usable_text_snippets"]) > 0
            dims_for_row = available_dimensions(has_caption=has_caption, has_context=has_context)
            if not dims_for_row:
                typer.echo(no_text_evidence_message(row), err=True)
                continue
            available_dims_by_fig[int(row["fig_index"])] = dims_for_row
            scorable_rows.append(row)
        target_rows = scorable_rows
        if not target_rows:
            continue
        target_indices = {int(row["fig_index"]) for row in target_rows}
        typer.echo(
            f"[paper {paper_index}/{total_papers} chunk {chunk_index}] agent scoring "
            f"{paper_rows[0]['paper_id']} figures {sorted(target_indices)} with {label}"
        )
        try:
            final_batch, audit_payload = run_agent_for_paper_chunk(
                paper_rows=paper_rows,
                target_rows=target_rows,
                feature_lookup=feature_lookup,
                context_lookup=context_lookup,
                available_dims_by_fig=available_dims_by_fig,
                prior_assessments=prior_assessments,
                vlm_provider=vlm_provider,
                vlm_model=vlm_model,
                llm_provider=llm_provider,
                llm_model=llm_model,
                max_retries=max_retries,
                include_paper_batch_context=include_paper_batch_context,
            )
        except Exception as exc:
            if len(target_rows) > 1:
                typer.echo(
                    f"Chunk failed for {paper_rows[0]['paper_id']} figures "
                    f"{sorted(target_indices)}; retrying each figure separately. "
                    f"Reason: {exc}",
                    err=True,
                )
                for row in target_rows:
                    row_fig_index = int(row["fig_index"])
                    single_dims = {row_fig_index: available_dims_by_fig[row_fig_index]}
                    try:
                        final_single, audit_payload = run_agent_for_paper_chunk(
                            paper_rows=paper_rows,
                            target_rows=[row],
                            feature_lookup=feature_lookup,
                            context_lookup=context_lookup,
                            available_dims_by_fig=single_dims,
                            prior_assessments=prior_assessments,
                            vlm_provider=vlm_provider,
                            vlm_model=vlm_model,
                            llm_provider=llm_provider,
                            llm_model=llm_model,
                            max_retries=max_retries,
                            include_paper_batch_context=include_paper_batch_context,
                        )
                    except Exception as fallback_exc:
                        error_records.append(
                            {
                                "paper_id": row["paper_id"],
                                "fig_index": row["fig_index"],
                                "source_type": "model",
                                "source_name": label,
                                "error": (
                                    f"Chunk failed: {exc}; "
                                    f"single-figure fallback failed: {fallback_exc}"
                                ),
                                "failed_at": datetime.now(UTC).isoformat(),
                            }
                        )
                        typer.echo(
                            f"Skipped {row['paper_id']} figure {row['fig_index']} "
                            f"after fallback due to error: {fallback_exc}",
                            err=True,
                        )
                        continue

                    item = final_single.figures[0]
                    scored = scored_row_from_final(row, label, item)
                    scored_rows.append(scored)
                    prior_assessments.append(
                        {
                            "fig_index": int(scored["fig_index"]),
                            "overall_score": scored.get("overall_score"),
                            "summary": scored.get("summary"),
                            "scores": {dim: scored.get(dim) for dim in DIMENSIONS},
                        }
                    )
                    audit_records.append(
                        {
                            "paper_id": paper_rows[0]["paper_id"],
                            "source_type": "model",
                            "source_name": label,
                            "prompt_mode": "agent_paper_batch_fallback_single",
                            "feature_coverage": {
                                "matched": (
                                    1
                                    if feature_status_for_row(row, feature_lookup) == "matched"
                                    else 0
                                ),
                                "total": 1,
                            },
                            "fallback_from_error": str(exc),
                            **audit_payload,
                        }
                    )
                continue

            for row in target_rows:
                error_records.append(
                    {
                        "paper_id": row["paper_id"],
                        "fig_index": row["fig_index"],
                        "source_type": "model",
                        "source_name": label,
                        "error": str(exc),
                        "failed_at": datetime.now(UTC).isoformat(),
                    }
                )
            skipped_figures = sorted(target_indices)
            typer.echo(
                f"Skipped {paper_rows[0]['paper_id']} figures {skipped_figures} "
                f"due to error: {exc}",
                err=True,
            )
            continue

        row_by_fig = {int(row["fig_index"]): row for row in target_rows}
        for item in sorted(final_batch.figures, key=lambda value: value.fig_index):
            row = row_by_fig[int(item.fig_index)]
            scored = scored_row_from_final(row, label, item)
            scored_rows.append(scored)
            prior_assessments.append(
                {
                    "fig_index": int(scored["fig_index"]),
                    "overall_score": scored.get("overall_score"),
                    "summary": scored.get("summary"),
                    "scores": {dim: scored.get(dim) for dim in DIMENSIONS},
                }
            )
        audit_records.append(
            {
                "paper_id": paper_rows[0]["paper_id"],
                "source_type": "model",
                "source_name": label,
                "prompt_mode": "agent_paper_batch",
                "feature_coverage": {
                    "matched": sum(
                        1
                        for row in target_rows
                        if feature_status_for_row(row, feature_lookup) == "matched"
                    ),
                    "total": len(target_rows),
                },
                **audit_payload,
            }
        )

    return scored_rows, audit_records, error_records


def run_agent_scoring(
    *,
    venue: str,
    year: int,
    stage: str = "clean",
    parquet: str | None,
    limit: int | None,
    paper_id: str | None,
    paper_limit: int | None,
    vlm_provider: str = "gemini",
    vlm_model: str,
    llm_provider: str = "gemini",
    llm_model: str,
    model_label: str | None,
    features_parquet: str | None,
    context_parquet: str | None = None,
    batch_mode: str = "figure",
    max_figures_per_call: int = 4,
    max_retries: int = 2,
    workers: int = 2,
    skip_upload: bool = False,
    manifest: str | None = None,
    output_parquet: str | None = None,
    output_jsonl: str | None = None,
) -> None:
    load_env()
    for provider_name, provider in [("vlm-provider", vlm_provider), ("llm-provider", llm_provider)]:
        validate_provider(provider, option_name=provider_name)
    if llm_provider == "qwen" and llm_model == vlm_model:
        llm_model = agent_llm_model(llm_provider, vlm_model)
    if stage not in {"clean", "raw"}:
        raise typer.BadParameter("--stage must be clean or raw.")
    if batch_mode not in {"figure", "paper"}:
        raise typer.BadParameter("--batch-mode must be figure or paper.")
    is_paper_batch = batch_mode == "paper"
    if max_figures_per_call < 1 or max_figures_per_call > 8:
        raise typer.BadParameter("--max-figures-per-call must be between 1 and 8.")
    if workers < 1:
        raise typer.BadParameter("--workers must be >= 1.")
    if not parquet and (not venue or not year):
        raise typer.BadParameter("Provide --parquet or both --venue and --year.")

    token = os.environ.get("HF_TOKEN", "").strip()
    hf_token = None if skip_upload and parquet else token
    figures = load_figures(parquet, venue, year, stage, hf_token)
    if venue and year:
        figures = figures.filter((pl.col("venue") == venue) & (pl.col("year") == year))
    figures = prepare_rows(figures, limit=limit, paper_id=paper_id)
    figures = filter_by_manifest(figures, manifest)
    if figures.is_empty():
        typer.echo("No rows to score.")
        raise typer.Exit()

    venue = str(figures["venue"][0])
    year = int(figures["year"][0])
    feature_lookup: dict[tuple[str, int], dict] = {}
    if features_parquet:
        feature_lookup = feature_lookup_map(load_feature_table(features_parquet))
        typer.echo(f"Loaded features from local parquet: {features_parquet}")
    elif token:
        try:
            feature_lookup = feature_lookup_map(
                load_features_from_hf(venue=venue, year=year, token=token)
            )
            typer.echo(
                f"Loaded features from HF: "
                f"processed/{venue}/{year}/features/figures_features.parquet"
            )
        except Exception as exc:
            typer.echo(
                f"Could not load HF features for {venue}/{year}; using feature_json=null. "
                f"Reason: {exc}",
                err=True,
            )
    else:
        typer.echo("No --features-parquet or HF_TOKEN; using feature_json=null.", err=True)

    if context_parquet:
        context_table = load_context_table(context_parquet)
        typer.echo(f"Loaded context from local parquet: {context_parquet}")
    elif token:
        try:
            context_table = load_context_from_hf(venue=venue, year=year, token=token)
            typer.echo(f"Loaded context from HF: processed/{venue}/{year}/figures_context.parquet")
        except Exception as exc:
            context_table = pl.DataFrame()
            typer.echo(
                f"Could not load HF context; context_consistency will be NA. Reason: {exc}",
                err=True,
            )
    else:
        context_table = pl.DataFrame()
    context_lookup = context_lookup_map(context_table)
    figures, skipped_scoreless = filter_rows_with_text_evidence(figures, context_lookup)
    if skipped_scoreless:
        typer.echo(f"Skipped {skipped_scoreless} rows without caption or usable body context.")
    figures = limit_to_first_papers(figures, paper_limit=paper_limit)
    if figures.is_empty():
        typer.echo("No rows to score after filtering.")
        raise typer.Exit()
    echo_feature_coverage(figures, feature_lookup)
    require_feature_coverage(figures, feature_lookup, required=False)

    default_label = f"agent:{vlm_provider}:{vlm_model}+{llm_provider}:{llm_model}:v1"
    if is_paper_batch:
        default_label = f"agent:{vlm_provider}:{vlm_model}+{llm_provider}:{llm_model}:paper-batch"
    label = model_label or default_label

    scored_rows: list[dict] = []
    audit_records: list[dict] = []
    error_records: list[dict] = []

    if batch_mode == "paper":
        paper_groups = figures.sort(["paper_id", "fig_index"]).partition_by(
            "paper_id", as_dict=False, maintain_order=True
        )
        total_papers = len(paper_groups)
        effective_workers = min(workers, total_papers) if total_papers else 1
        if effective_workers > 1:
            typer.echo(
                f"Agent paper-batch scoring {total_papers} papers "
                f"with {effective_workers} workers..."
            )

        paper_jobs = [
            {
                "paper_index": paper_index,
                "total_papers": total_papers,
                "paper_df": paper_df,
                "feature_lookup": feature_lookup,
                "context_lookup": context_lookup,
                "label": label,
                "max_figures_per_call": max_figures_per_call,
                "vlm_provider": vlm_provider,
                "vlm_model": vlm_model,
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "max_retries": max_retries,
                "include_paper_batch_context": True,
            }
            for paper_index, paper_df in enumerate(paper_groups, start=1)
        ]

        if effective_workers == 1:
            for job in paper_jobs:
                paper_scored, paper_audit, paper_errors = run_agent_for_paper(**job)
                scored_rows.extend(paper_scored)
                _append_scored_jsonl(output_jsonl, paper_scored)
                audit_records.extend(paper_audit)
                error_records.extend(paper_errors)
        else:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = [executor.submit(run_agent_for_paper, **job) for job in paper_jobs]
                for future in as_completed(futures):
                    paper_scored, paper_audit, paper_errors = future.result()
                    scored_rows.extend(paper_scored)
                    _append_scored_jsonl(output_jsonl, paper_scored)
                    audit_records.extend(paper_audit)
                    error_records.extend(paper_errors)

        error_audit = write_error_audit(
            error_records,
            provider="agent",
            venue=venue,
            year=year,
            model_label=label,
        )
        if error_audit is not None:
            typer.echo(f"Error log written to {error_audit}")

        if not scored_rows:
            typer.echo("No successful rows were scored.")
            raise typer.Exit(code=1)

        fresh = pl.DataFrame(scored_rows, infer_schema_length=len(scored_rows)).with_columns(
            pl.col("year").cast(pl.Int16),
            pl.col("fig_index").cast(pl.Int32),
            *[pl.col(dim).cast(pl.Float32) for dim in DIMENSIONS + ["overall_score"]],
        )
        audit = write_audit(
            audit_records,
            provider="agent",
            venue=venue,
            year=year,
            model_label=label,
        )
        typer.echo(f"Audit log written to {audit}")

        existing = load_existing_results(venue=venue, year=year, token=hf_token)
        merged = merge_results(existing, fresh)
        if output_parquet:
            local_parquet = Path(output_parquet)
            local_parquet.parent.mkdir(parents=True, exist_ok=True)
        else:
            local_parquet = audit.with_suffix(".parquet")
        merged.write_parquet(local_parquet)
        typer.echo(f"Merged parquet written to {local_parquet}")

        if skip_upload:
            typer.echo("Skipped HF upload.")
            return
        if not hf_token:
            raise typer.BadParameter("HF_TOKEN is required for upload.")
        upload_results(merged, venue=venue, year=year, token=hf_token)
        typer.echo(f"Uploaded processed/{venue}/{year}/results.parquet to {REPO_ID}")
        return

    paper_groups = figures.sort(["paper_id", "fig_index"]).partition_by(
        "paper_id", as_dict=False, maintain_order=True
    )
    total_rows = len(figures)
    row_index = 0
    for paper_df in paper_groups:
        paper_rows = list(paper_df.sort("fig_index").iter_rows(named=True))
        for row in paper_rows:
            row_index += 1
            typer.echo(
                f"[{row_index}/{total_rows}] agent scoring {row['paper_id']} "
                f"figure {row['fig_index']} with {label}"
            )
            context_payload = build_context_payload(context_lookup.get(context_key(row), []))
            has_caption = bool(str(row.get("caption") or "").strip())
            has_context = int(context_payload["usable_text_snippets"]) > 0
            dims_for_row = available_dimensions(has_caption=has_caption, has_context=has_context)
            if not dims_for_row:
                typer.echo(no_text_evidence_message(row), err=True)
                continue
            fig_index = int(row["fig_index"])
            try:
                final_batch, audit_payload = run_agent_for_paper_chunk(
                    paper_rows=paper_rows,
                    target_rows=[row],
                    feature_lookup=feature_lookup,
                    context_lookup=context_lookup,
                    available_dims_by_fig={fig_index: dims_for_row},
                    prior_assessments=[],
                    vlm_provider=vlm_provider,
                    vlm_model=vlm_model,
                    llm_provider=llm_provider,
                    llm_model=llm_model,
                    max_retries=max_retries,
                    include_paper_batch_context=False,
                )
            except Exception as exc:
                error_records.append(
                    {
                        "paper_id": row["paper_id"],
                        "fig_index": row["fig_index"],
                        "source_type": "model",
                        "source_name": label,
                        "error": str(exc),
                        "failed_at": datetime.now(UTC).isoformat(),
                    }
                )
                typer.echo(
                    f"Skipped {row['paper_id']} figure {row['fig_index']} due to error: {exc}",
                    err=True,
                )
                continue

            final = final_batch.figures[0]
            scored = scored_row_from_final(row, label, final)
            scored_rows.append(scored)
            _append_scored_jsonl(output_jsonl, [scored])
            audit_records.append(
                {
                    "paper_id": row["paper_id"],
                    "fig_index": row["fig_index"],
                    "source_type": "model",
                    "source_name": label,
                    "feature_status": feature_status_for_row(row, feature_lookup),
                    **audit_payload,
                }
            )

    error_audit = write_error_audit(
        error_records,
        provider="agent",
        venue=venue,
        year=year,
        model_label=label,
    )
    if error_audit is not None:
        typer.echo(f"Error log written to {error_audit}")

    if not scored_rows:
        typer.echo("No successful rows were scored.")
        raise typer.Exit(code=1)

    fresh = pl.DataFrame(scored_rows).with_columns(
        pl.col("year").cast(pl.Int16),
        pl.col("fig_index").cast(pl.Int32),
        *[pl.col(dim).cast(pl.Float32) for dim in DIMENSIONS + ["overall_score"]],
    )

    audit = write_audit(
        audit_records,
        provider="agent",
        venue=venue,
        year=year,
        model_label=label,
    )
    typer.echo(f"Audit log written to {audit}")

    existing = load_existing_results(venue=venue, year=year, token=hf_token)
    merged = merge_results(existing, fresh)

    if output_parquet:
        local_parquet = Path(output_parquet)
        local_parquet.parent.mkdir(parents=True, exist_ok=True)
    else:
        local_parquet = audit.with_suffix(".parquet")
    merged.write_parquet(local_parquet)
    typer.echo(f"Merged parquet written to {local_parquet}")

    if skip_upload:
        typer.echo("Skipped HF upload.")
        return
    if not hf_token:
        raise typer.BadParameter("HF_TOKEN is required for upload.")
    upload_results(merged, venue=venue, year=year, token=hf_token)
    typer.echo(f"Uploaded processed/{venue}/{year}/results.parquet to {REPO_ID}")
