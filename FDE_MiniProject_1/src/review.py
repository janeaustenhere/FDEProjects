from __future__ import annotations

from .models import ClassificationResult, ProcessingStatus, ReviewSource
from .taxonomy import BodyArea, PrimaryReason, SubReason


def apply_human_review(
    result: ClassificationResult,
    primary: str,
    sub_reason: str | None,
    body_area: str | None,
    discovered_category: str | None = None,
    discovered_reason: str | None = None,
) -> ClassificationResult:
    return ClassificationResult(
        **result.model_dump(exclude={"primary_reason", "sub_reason", "body_area", "normalized_summary", "discovered_category", "discovered_reason", "confidence", "needs_human_review", "processing_status", "failure_reason", "review_source", "secondary_reasons"}),
        primary_reason=PrimaryReason(primary),
        sub_reason=SubReason(sub_reason) if sub_reason else None,
        body_area=BodyArea(body_area) if body_area else None,
        secondary_reasons=[],
        normalized_summary=f"Human reviewer classified this return as {primary.replace('_', ' ').lower()}.",
        discovered_category=discovered_category if primary == "OTHER_KNOWN" else None,
        discovered_reason=discovered_reason if primary == "OTHER_KNOWN" else None,
        confidence=1.0,
        needs_human_review=False,
        processing_status=ProcessingStatus.ACCEPTED,
        failure_reason=None,
        review_source=ReviewSource.HUMAN,
    )
