from __future__ import annotations

from .models import ClassificationResult
from .preprocessing import contains_multiple_issues
from .taxonomy import PrimaryReason


def should_route(result: ClassificationResult, threshold: float) -> tuple[bool, str]:
    if result.confidence < threshold:
        return True, f"Confidence {result.confidence:.2f} is below threshold {threshold:.2f}"
    if result.primary_reason == PrimaryReason.UNCERTAIN:
        return True, "Bulk model marked the comment uncertain"
    if result.needs_human_review:
        return True, "Bulk model requested human review"
    if result.primary_reason == PrimaryReason.OTHER_KNOWN:
        return True, "Newly discovered reason requires verification"
    if contains_multiple_issues(result.return_comment):
        return True, "Comment may contain multiple issues"
    return False, "High-confidence classification"
