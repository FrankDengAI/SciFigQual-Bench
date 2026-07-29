from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import typer
from feature_inputs import feature_lookup_map, load_feature_table
from io_utils import (
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
from paper_inputs import (
    build_context_payload,
    context_key,
    context_lookup_map,
    filter_rows_with_text_evidence,
    format_paper_payload,
    load_context_table,
    paper_prompt_payload,
)
from providers import generate_paper_batch_with_provider
from provider_registry import validate_provider
from schema import (
    DIMENSIONS,
    PaperAssessment,
    available_dimensions,
    compute_overall,
    validate_paper_assessment,
)

PROMPT_PATHS = {
    "baseline": Path(__file__).resolve().parent / "prompts" / "baseline_paper_batch.md",
    "with_features": Path(__file__).resolve().parent / "prompts" / "with_features_paper_batch.md",
}


def no_text_evidence_message(row: dict) -> str:
    return (
        f"Skipped {row['paper_id']} figure {row['fig_index']} because it has neither "
        "caption nor usable context."
    )


def chunks(values: list[dict], size: int) -> list[list[dict]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def call_with_retries(fn, max_retries: int) -> PaperAssessment:
    if max_retries < 1:
        raise typer.BadParameter("--max-retries must be >= 1.")
    last_error: Exception | None = None
    for _ in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def scored_row_from_assessment(row: dict, label: str, item) -> dict:
    dim_scores = {
        dim: (getattr(item, dim).score if getattr(item, dim) is not None else None)
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
        "summary": item.summary,
        "suggestion": item.suggestion,
        "visual_clarity_reason": item.visual_clarity.reason,
        "structure_layout_reason": item.structure_layout.reason,
        "caption_consistency_reason": (
            item.caption_consistency.reason if item.caption_consistency is not None else None
        ),
        "context_consistency_reason": (
            item.context_consistency.reason if item.context_consistency is not None else None
        ),
        "misleading_risk_reason": item.misleading_risk.reason,
        "annotated_at": datetime.now(UTC).isoformat(),
    }


def prior_summary(row: dict) -> dict:
    return {
        "fig_index": int(row["fig_index"]),
        "overall_score": row.get("overall_score"),
        "summary": row.get("summary"),
        "scores": {dim: row.get(dim) for dim in DIMENSIONS},
    }


def _append_scored_jsonl(output_jsonl: str | None, rows: list[dict]) -> None:
    if not output_jsonl:
        return
    for row in rows:
        append_result_jsonl(output_jsonl, row)


def run_one_paper(
    *,
    paper_index: int,
    total_papers: int,
    paper_df: pl.DataFrame,
    feature_lookup: dict,
    context_lookup: dict[tuple[str, int], list[dict]],
    label: str,
    prompt: str,
    prompt_mode: str,
    provider: str,
    model: str,
    max_figures_per_call: int,
    max_retries: int,
    include_paper_batch_context: bool,
) -> tuple[list[dict], list[dict], list[dict]]:
    scored_rows: list[dict] = []
    audit_records: list[dict] = []
    error_records: list[dict] = []

    paper_rows = list(paper_df.sort("fig_index").iter_rows(named=True))
    prior_assessments: list[dict] = []
    for chunk_index, target_rows in enumerate(chunks(paper_rows, max_figures_per_call), start=1):
        available_dims_by_fig: dict[int, list[str]] = {}
        scorable_rows: list[dict] = []
        for row in target_rows:
            context_payload = build_context_payload(context_lookup.get(context_key(row), []))
            has_caption = bool(str(row.get("caption") or "").strip())
            has_context = int(context_payload["usable_text_snippets"]) > 0
            dims_for_row = available_dimensions(
                has_caption=has_caption,
                has_context=has_context,
            )
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
            f"[paper {paper_index}/{total_papers} chunk {chunk_index}] scoring "
            f"{paper_rows[0]['paper_id']} figures {sorted(target_indices)} with {label}"
        )
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
        try:
            payload_text = f"paper_batch_json: {format_paper_payload(payload)}"
            assessment = call_with_retries(
                lambda target_rows=target_rows, payload_text=payload_text: (
                    generate_paper_batch_with_provider(
                        provider=provider,
                        rows=target_rows,
                        model=model,
                        prompt=prompt,
                        response_model=PaperAssessment,
                        user_payload_text=payload_text,
                    )
                ),
                max_retries=max_retries,
            )
            validate_paper_assessment(assessment, target_indices, available_dims_by_fig)
        except Exception as exc:
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
        for item in sorted(assessment.figures, key=lambda value: value.fig_index):
            row = row_by_fig[int(item.fig_index)]
            scored = scored_row_from_assessment(row, label, item)
            scored_rows.append(scored)
            prior_assessments.append(prior_summary(scored))
            audit_records.append(
                {
                    "paper_id": row["paper_id"],
                    "fig_index": row["fig_index"],
                    "source_type": "model",
                    "source_name": label,
                    "prompt_mode": f"{prompt_mode}_paper_batch",
                    "paper_batch_payload": payload,
                    "assessment": item.model_dump(),
                }
            )

    return scored_rows, audit_records, error_records


def run_paper_batch_scoring(
    *,
    provider: str = "gemini",
    venue: str,
    year: int,
    stage: str = "clean",
    parquet: str | None,
    limit: int | None,
    paper_id: str | None,
    paper_limit: int | None,
    model: str,
    model_label: str | None,
    features_parquet: str | None,
    prompt_mode: str = "baseline",
    context_parquet: str | None = None,
    max_figures_per_call: int = 5,
    max_retries: int = 2,
    workers: int = 2,
    skip_upload: bool = False,
    include_paper_batch_context: bool = True,
    manifest: str | None = None,
    output_parquet: str | None = None,
    output_jsonl: str | None = None,
) -> None:
    load_env()
    validate_provider(provider)
    if stage not in {"clean", "raw"}:
        raise typer.BadParameter("--stage must be clean or raw.")
    if prompt_mode not in {"baseline", "with_features"}:
        raise typer.BadParameter("--prompt-mode must be baseline or with_features.")
    if max_figures_per_call < 1:
        raise typer.BadParameter("--max-figures-per-call must be >= 1.")
    if max_figures_per_call > 8:
        raise typer.BadParameter("--max-figures-per-call must be <= 8.")
    if workers < 1:
        raise typer.BadParameter("--workers must be >= 1.")

    token = os.environ.get("HF_TOKEN", "").strip()
    hf_token = None if skip_upload and parquet else token
    figures = load_figures(parquet, venue, year, stage, hf_token)
    if venue and year:
        figures = figures.filter((pl.col("venue") == venue) & (pl.col("year") == year))
    figures = prepare_rows(figures, limit=limit, paper_id=paper_id).sort(["paper_id", "fig_index"])
    figures = filter_by_manifest(figures, manifest)
    if figures.is_empty():
        typer.echo("No rows to score.")
        raise typer.Exit()

    venue = str(figures["venue"][0])
    year = int(figures["year"][0])
    label = model_label or f"{provider}:{model}:{prompt_mode}:paper-batch"

    feature_lookup = {}
    if prompt_mode == "with_features" and features_parquet:
        feature_lookup = feature_lookup_map(load_feature_table(features_parquet))
        typer.echo(f"Loaded features from local parquet: {features_parquet}")
    elif prompt_mode == "with_features" and token:
        try:
            feature_lookup = feature_lookup_map(load_features_from_hf(venue, year, token))
            typer.echo(
                f"Loaded features from HF: processed/{venue}/{year}/features/"
                "figures_features.parquet"
            )
        except Exception as exc:
            typer.echo(
                f"Could not load HF features; continuing without features. Reason: {exc}", err=True
            )

    if context_parquet:
        context_table = load_context_table(context_parquet)
        typer.echo(f"Loaded context from local parquet: {context_parquet}")
    elif token:
        try:
            context_table = load_context_from_hf(venue, year, token)
            typer.echo(f"Loaded context from HF: processed/{venue}/{year}/figures_context.parquet")
        except Exception as exc:
            context_table = pl.DataFrame()
            typer.echo(
                f"Could not load HF context; continuing without body context. Reason: {exc}",
                err=True,
            )
    else:
        context_table = pl.DataFrame()
    context_lookup = context_lookup_map(context_table)
    figures, skipped_scoreless = filter_rows_with_text_evidence(figures, context_lookup)
    if skipped_scoreless:
        typer.echo(f"Skipped {skipped_scoreless} rows without caption or usable body context.")
    figures = limit_to_first_papers(figures, paper_limit=paper_limit).sort(
        ["paper_id", "fig_index"]
    )
    if figures.is_empty():
        typer.echo("No rows to score after filtering.")
        raise typer.Exit()

    prompt = PROMPT_PATHS[prompt_mode].read_text(encoding="utf-8")
    scored_rows: list[dict] = []
    audit_records: list[dict] = []
    error_records: list[dict] = []

    paper_groups = figures.partition_by("paper_id", as_dict=False, maintain_order=True)
    total_papers = len(paper_groups)
    effective_workers = min(workers, total_papers) if total_papers else 1
    if effective_workers > 1:
        typer.echo(f"Paper-batch scoring {total_papers} papers with {effective_workers} workers...")

    paper_jobs = [
        {
            "paper_index": paper_index,
            "total_papers": total_papers,
            "paper_df": paper_df,
            "feature_lookup": feature_lookup,
            "context_lookup": context_lookup,
            "label": label,
            "prompt": prompt,
            "prompt_mode": prompt_mode,
            "provider": provider,
            "model": model,
            "max_figures_per_call": max_figures_per_call,
            "max_retries": max_retries,
            "include_paper_batch_context": include_paper_batch_context,
        }
        for paper_index, paper_df in enumerate(paper_groups, start=1)
    ]
    if effective_workers == 1:
        for job in paper_jobs:
            paper_scored, paper_audit, paper_errors = run_one_paper(**job)
            scored_rows.extend(paper_scored)
            _append_scored_jsonl(output_jsonl, paper_scored)
            audit_records.extend(paper_audit)
            error_records.extend(paper_errors)
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(run_one_paper, **job) for job in paper_jobs]
            for future in as_completed(futures):
                paper_scored, paper_audit, paper_errors = future.result()
                scored_rows.extend(paper_scored)
                _append_scored_jsonl(output_jsonl, paper_scored)
                audit_records.extend(paper_audit)
                error_records.extend(paper_errors)

    error_audit = write_error_audit(error_records, "paper_batch", venue, year, label)
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
    audit = write_audit(audit_records, "paper_batch", venue, year, label)
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
