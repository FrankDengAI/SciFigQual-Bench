from __future__ import annotations

from pydantic import BaseModel, Field

# Keep in sync with src/cs64/config/__init__.py DIMENSIONS.
# Cannot import from cs64 here due to sys.path injection ordering.
DIMENSIONS = [
    "visual_clarity",
    "structure_layout",
    "caption_consistency",
    "context_consistency",
    "misleading_risk",
]


class DimensionAssessment(BaseModel):
    score: int
    confidence: int
    reason: str


class AgentFinalAssessment(BaseModel):
    visual_clarity: DimensionAssessment
    structure_layout: DimensionAssessment
    caption_consistency: DimensionAssessment | None = None
    context_consistency: DimensionAssessment | None = None
    misleading_risk: DimensionAssessment
    summary: str
    suggestion: str


class PaperAgentFinalItem(AgentFinalAssessment):
    fig_index: int


class PaperAgentFinalAssessment(BaseModel):
    figures: list[PaperAgentFinalItem]


class V5VisualScores(BaseModel):
    visual_clarity: DimensionAssessment
    structure_layout: DimensionAssessment


class V5ImageSummary(BaseModel):
    figure_type: str
    main_content: str
    visible_takeaway: str | None = None
    visible_panels: list[str] = Field(default_factory=list)
    visible_metrics: list[str] = Field(default_factory=list)
    visible_compared_groups: list[str] = Field(default_factory=list)
    visible_axes_or_units: list[str] = Field(default_factory=list)
    visible_legend_or_label_notes: str | None = None
    uncertain_visual_details: list[str] = Field(default_factory=list)


class V5ImageImpliedCaption(BaseModel):
    expected_caption_summary: str
    caption_should_mention: list[str] = Field(default_factory=list)
    caption_not_needed: list[str] = Field(default_factory=list)


class V5ImageImpliedContext(BaseModel):
    expected_context_summary: str
    context_should_support: list[str] = Field(default_factory=list)


class V5VisualTrustNotes(BaseModel):
    trustworthy_signals: list[str] = Field(default_factory=list)
    risk_signals: list[str] = Field(default_factory=list)
    possible_misleading_use: list[str] = Field(default_factory=list)


class V5VisualMisleadingRisk(BaseModel):
    intrinsic_risk_band: str
    intrinsic_risk_severity: str
    reason: str


class V5VlmEvidenceItem(BaseModel):
    fig_index: int
    visual_scores: V5VisualScores
    image_summary: V5ImageSummary
    image_implied_caption: V5ImageImpliedCaption
    image_implied_context: V5ImageImpliedContext
    visual_trust_notes: V5VisualTrustNotes
    visual_misleading_risk: V5VisualMisleadingRisk


class V5PaperVlmEvidenceReport(BaseModel):
    figures: list[V5VlmEvidenceItem]


class V5CaptionSummary(BaseModel):
    raw_caption_or_short_summary: str | None = None
    caption_claims: list[str] = Field(default_factory=list)
    mentioned_metrics: list[str] = Field(default_factory=list)
    mentioned_compared_groups: list[str] = Field(default_factory=list)
    mentioned_datasets_or_conditions: list[str] = Field(default_factory=list)
    mentioned_panels: list[str] = Field(default_factory=list)
    caption_expected_image: str | None = None
    caption_missing_likely_requirements: list[str] = Field(default_factory=list)


class V5ContextSummary(BaseModel):
    has_usable_context: bool
    raw_context_if_short_or_summary: str | None = None
    context_claims: list[str] = Field(default_factory=list)
    context_support_level: str
    context_expected_image: str | None = None
    specific_entities_metrics_or_trends: list[str] = Field(default_factory=list)


class V5FeatureSummary(BaseModel):
    ocr_quality: str | None = None
    text_density: str | None = None
    caption_ocr_overlap: str | None = None
    numeric_overlap: str | None = None
    feature_notes: list[str] = Field(default_factory=list)


class V5TextImpliedImage(BaseModel):
    expected_figure_type: str | None = None
    expected_visible_content: list[str] = Field(default_factory=list)
    expected_but_unspecified: list[str] = Field(default_factory=list)


class V5TextMisleadingAdjustment(BaseModel):
    severity: str
    reason: str


class V5RuleFlags(BaseModel):
    is_result_or_comparison_figure: bool
    caption_mentions_metric: bool
    caption_identifies_compared_groups: bool
    context_is_bare_reference: bool
    context_has_specific_support: bool
    caption_score_cap: int | None = None
    context_score_cap: int | None = None
    misleading_score_cap: int | None = None
    flag_rationale: list[str] = Field(default_factory=list)


class V5LlmEvidenceItem(BaseModel):
    fig_index: int
    caption_summary: V5CaptionSummary
    context_summary: V5ContextSummary
    feature_summary: V5FeatureSummary
    caption_implied_image: V5TextImpliedImage
    context_implied_image: V5TextImpliedImage
    text_misleading_adjustment: V5TextMisleadingAdjustment
    rule_flags: V5RuleFlags


class V5PaperLlmEvidenceReport(BaseModel):
    figures: list[V5LlmEvidenceItem]


def available_dimensions(*, has_caption: bool, has_context: bool) -> list[str]:
    if not has_caption and not has_context:
        return []
    dims = list(DIMENSIONS)
    if not has_caption:
        dims.remove("caption_consistency")
    if not has_context:
        dims.remove("context_consistency")
    return dims


def validate_dimension(dim: str, assessment: DimensionAssessment | None, *, required: bool) -> None:
    if assessment is None:
        if required:
            raise ValueError(f"{dim} is required but missing")
        return
    if not 1 <= int(assessment.score) <= 10:
        raise ValueError(f"{dim} score out of range: {assessment.score}")
    if not 1 <= int(assessment.confidence) <= 10:
        raise ValueError(f"{dim} confidence out of range: {assessment.confidence}")


def validate_final(
    assessment: AgentFinalAssessment, available_dims: list[str] | None = None
) -> None:
    required_dims = set(available_dims or DIMENSIONS)
    for dim in DIMENSIONS:
        validate_dimension(dim, getattr(assessment, dim), required=dim in required_dims)


def validate_paper_final(
    assessment: PaperAgentFinalAssessment,
    expected_fig_indices: set[int],
    available_dims_by_fig: dict[int, list[str]] | None = None,
) -> None:
    seen: set[int] = set()
    for item in assessment.figures:
        fig_index = int(item.fig_index)
        if fig_index in seen:
            raise ValueError(f"Duplicate fig_index in paper agent response: {fig_index}")
        seen.add(fig_index)
        validate_final(
            item,
            None if available_dims_by_fig is None else available_dims_by_fig.get(fig_index),
        )
    missing = sorted(expected_fig_indices - seen)
    extra = sorted(seen - expected_fig_indices)
    if missing:
        raise ValueError(f"Paper agent response missing target figures: {missing}")
    if extra:
        raise ValueError(f"Paper agent response included non-target figures: {extra}")


def compute_overall(payload: dict[str, int | None]) -> float | None:
    values = [value for value in payload.values() if value is not None]
    return round(sum(values) / len(values), 1) if values else None
