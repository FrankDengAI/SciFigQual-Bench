from __future__ import annotations

import typer


def run_scoring(
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
    max_retries: int = 2,
    skip_upload: bool,
    prompt_mode: str,
    features_parquet: str | None,
    context_parquet: str | None = None,
    batch_mode: str = "figure",
    max_figures_per_call: int = 5,
    workers: int = 4,
    manifest: str | None = None,
    output_parquet: str | None = None,
    output_jsonl: str | None = None,
) -> None:
    if prompt_mode not in {"baseline", "with_features"}:
        raise typer.BadParameter("--prompt-mode must be baseline or with_features.")
    if batch_mode not in {"figure", "paper"}:
        raise typer.BadParameter("--batch-mode must be figure or paper.")
    if max_figures_per_call < 1 or max_figures_per_call > 8:
        raise typer.BadParameter("--max-figures-per-call must be between 1 and 8.")

    from paper_runner import run_paper_batch_scoring

    run_paper_batch_scoring(
        provider=provider,
        venue=venue,
        year=year,
        stage=stage,
        parquet=parquet,
        limit=limit,
        paper_id=paper_id,
        paper_limit=paper_limit,
        model=model,
        model_label=model_label,
        prompt_mode=prompt_mode,
        features_parquet=features_parquet,
        context_parquet=context_parquet,
        max_figures_per_call=max_figures_per_call if batch_mode == "paper" else 1,
        max_retries=max_retries,
        skip_upload=skip_upload,
        workers=workers,
        include_paper_batch_context=batch_mode == "paper",
        manifest=manifest,
        output_parquet=output_parquet,
        output_jsonl=output_jsonl,
    )
