from __future__ import annotations

import re

from .taxonomy import PrimaryReason, SubReason

DEFAULT_VENDOR = "Unknown Vendor"
OTHER_REASON_ALIASES = frozenset({"other", "others", "other reason"})
CLASSIFICATION_SOURCE_AI = "AI interpretation of Other"
CLASSIFICATION_SOURCE_STRUCTURED = "Existing return reason"
REASON_MATCH_EXISTING = "Existing return reason"
REASON_MATCH_NEW = "New reason"

STRUCTURED_REASON_MAP: dict[str, tuple[PrimaryReason, SubReason | None]] = {
    "size issue": (PrimaryReason.SIZE_INFORMATION, SubReason.SIZE_INFORMATION_OTHER),
    "sizing issue": (PrimaryReason.SIZE_INFORMATION, SubReason.SIZE_INFORMATION_OTHER),
    "fit issue": (PrimaryReason.FIT, SubReason.FIT_OTHER),
    "defective piece": (PrimaryReason.DAMAGED_OR_DEFECTIVE, SubReason.DEFECT_OTHER),
    "defective pieces": (PrimaryReason.DAMAGED_OR_DEFECTIVE, SubReason.DEFECT_OTHER),
    "damaged": (PrimaryReason.DAMAGED_OR_DEFECTIVE, SubReason.DEFECT_OTHER),
    "not delivered": (PrimaryReason.DELIVERY_OR_PACKAGING, SubReason.NOT_DELIVERED),
    "late delivery": (PrimaryReason.DELIVERY_OR_PACKAGING, SubReason.LATE_DELIVERY),
    "wrong item": (PrimaryReason.WRONG_ITEM, SubReason.WRONG_ITEM_OTHER),
    "quality issue": (PrimaryReason.QUALITY, SubReason.QUALITY_OTHER),
    "product mismatch": (PrimaryReason.PRODUCT_MISMATCH, SubReason.PRODUCT_MISMATCH_OTHER),
    "changed mind": (PrimaryReason.CUSTOMER_PREFERENCE, SubReason.CHANGED_MIND),
}


def normalise_reason(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().casefold()).strip()


def is_other_reason(value: str) -> bool:
    return normalise_reason(value) in OTHER_REASON_ALIASES


def structured_reason_mapping(value: str) -> tuple[PrimaryReason, SubReason | None] | None:
    return STRUCTURED_REASON_MAP.get(normalise_reason(value))


def business_category_for(
    primary: PrimaryReason | str,
    sub_reason: SubReason | str | None = None,
    discovered_category: str | None = None,
) -> str:
    primary_reason = PrimaryReason(primary)
    sub = SubReason(sub_reason) if sub_reason else None
    if primary_reason in {PrimaryReason.FIT, PrimaryReason.SIZE_INFORMATION}:
        return "Size Issue"
    if primary_reason == PrimaryReason.DAMAGED_OR_DEFECTIVE:
        return "Defective Pieces"
    if primary_reason == PrimaryReason.WRONG_ITEM:
        return "Wrong Item"
    if primary_reason == PrimaryReason.DELIVERY_OR_PACKAGING and sub == SubReason.NOT_DELIVERED:
        return "Not Delivered"
    if primary_reason == PrimaryReason.DELIVERY_OR_PACKAGING and sub == SubReason.LATE_DELIVERY:
        return "Late Delivery"
    if primary_reason == PrimaryReason.OTHER_KNOWN:
        return discovered_category or "Other Known"
    return primary_reason.value.replace("_", " ").title()
