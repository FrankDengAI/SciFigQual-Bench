from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from cs64.config import DIMENSIONS

__all__ = ["DIMENSIONS"]


class DimensionAssessment(BaseModel):
    score: int
    confidence: int
    reason: str


class FigureAssessment(BaseModel):
    visual_clarity: DimensionAssessment
    structure_layout: DimensionAssessment
    caption_consistency: DimensionAssessment | None = None
    context_consistency: DimensionAssessment | None = None
    misleading_risk: DimensionAssessment
    summary: str
    suggestion: str


class PaperFigureAssessment(FigureAssessment):
    fig_index: int


class PaperAssessment(BaseModel):
    figures: list[PaperFigureAssessment]


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def compute_overall(payload: dict[str, int | None]) -> float | None:
    values = [value for value in payload.values() if value is not None]
    return round(sum(values) / len(values), 1) if values else None


def available_dimensions(*, has_caption: bool, has_context: bool) -> list[str]:
    if not has_caption and not has_context:
        return []
    dims = list(DIMENSIONS)
    if not has_caption:
        dims.remove("caption_consistency")
    if not has_context:
        dims.remove("context_consistency")
    return dims


def validate_dimension(dim: str, value: DimensionAssessment | None, *, required: bool) -> None:
    if value is None:
        if required:
            raise ValueError(f"{dim} is required but missing")
        return
    score = value.score
    confidence = value.confidence
    if not 1 <= int(score) <= 10:
        raise ValueError(f"{dim} score out of range: {score}")
    if not 1 <= int(confidence) <= 10:
        raise ValueError(f"{dim} confidence out of range: {confidence}")


def validate_scores(
    assessment: FigureAssessment,
    available_dims: list[str] | None = None,
) -> None:
    required_dims = set(available_dims or DIMENSIONS)
    for dim in DIMENSIONS:
        validate_dimension(dim, getattr(assessment, dim), required=dim in required_dims)


def validate_paper_assessment(
    assessment: PaperAssessment,
    expected_fig_indices: set[int],
    available_dims_by_fig: dict[int, list[str]] | None = None,
) -> None:
    seen: set[int] = set()
    for item in assessment.figures:
        if int(item.fig_index) in seen:
            raise ValueError(f"Duplicate fig_index in paper response: {item.fig_index}")
        seen.add(int(item.fig_index))
        validate_scores(
            item,
            None
            if available_dims_by_fig is None
            else available_dims_by_fig.get(int(item.fig_index)),
        )
    missing = sorted(expected_fig_indices - seen)
    extra = sorted(seen - expected_fig_indices)
    if missing:
        raise ValueError(f"Paper response missing target figures: {missing}")
    if extra:
        raise ValueError(f"Paper response included non-target figures: {extra}")
