from __future__ import annotations

from .models import ClassificationResult, ProcessingStatus, ReturnRecord
from .reason_registry import structured_reason_mapping
from .taxonomy import PrimaryReason


def structured_reason_result(record: ReturnRecord) -> ClassificationResult:
    mapped = structured_reason_mapping(record.return_reason)
    if mapped is None:
        primary, sub = PrimaryReason.OTHER_KNOWN, None
        discovered_category = record.return_reason.strip() or "Unspecified"
        discovered_reason = "Customer selected this structured return reason."
    else:
        primary, sub = mapped
        discovered_category = None
        discovered_reason = None
    return ClassificationResult(
        return_id=record.return_id,
        sku_id=record.sku_id,
        category=record.category,
        return_comment=record.return_comment,
        source_return_reason=record.return_reason,
        vendor=record.vendor,
        primary_reason=primary,
        sub_reason=sub,
        normalized_summary=f"Customer selected {record.return_reason.strip()} as the return reason.",
        discovered_category=discovered_category,
        discovered_reason=discovered_reason,
        confidence=1.0,
        needs_human_review=False,
        model_name="structured-input",
        processing_status=ProcessingStatus.ACCEPTED,
    )
