import pytest
from pydantic import ValidationError

from src.models import ClassificationPayload


def test_valid_fit_payload():
    result = ClassificationPayload(primary_reason="FIT", sub_reason="TOO_TIGHT", body_area="CHEST", normalized_summary="Too tight at chest.", confidence=.9, needs_human_review=False)
    assert result.sub_reason.value == "TOO_TIGHT"


def test_invalid_primary_sub_pair():
    with pytest.raises(ValidationError):
        ClassificationPayload(primary_reason="FIT", sub_reason="FABRIC_QUALITY", normalized_summary="Bad pair.", confidence=.9, needs_human_review=False)


def test_uncertain_requires_review():
    with pytest.raises(ValidationError):
        ClassificationPayload(primary_reason="UNCERTAIN", normalized_summary="Unknown.", confidence=.2, needs_human_review=False)


def test_extra_model_fields_are_rejected():
    with pytest.raises(ValidationError):
        ClassificationPayload(
            primary_reason="FIT", sub_reason="TOO_TIGHT", normalized_summary="Tight.",
            confidence=.9, needs_human_review=False, invented_field="not allowed",
        )


def test_discovered_fields_are_only_valid_for_other_known():
    with pytest.raises(ValidationError, match="only valid for OTHER_KNOWN"):
        ClassificationPayload(
            primary_reason="FIT", sub_reason="TOO_TIGHT", normalized_summary="Tight.",
            discovered_category="Unexpected", discovered_reason="Unexpected.",
            confidence=.9, needs_human_review=False,
        )


def test_uncertain_cannot_be_a_secondary_reason():
    with pytest.raises(ValidationError, match="cannot be secondary"):
        ClassificationPayload(
            primary_reason="FIT", sub_reason="TOO_TIGHT",
            secondary_reasons=[{"primary_reason": "UNCERTAIN"}],
            normalized_summary="Tight with another unclear issue.", confidence=.8,
            needs_human_review=False,
        )


def test_duplicate_secondary_reason_is_rejected():
    with pytest.raises(ValidationError, match="Duplicate secondary"):
        ClassificationPayload(
            primary_reason="FIT", sub_reason="TOO_TIGHT",
            secondary_reasons=[
                {"primary_reason": "QUALITY", "sub_reason": "STITCHING"},
                {"primary_reason": "QUALITY", "sub_reason": "STITCHING"},
            ],
            normalized_summary="Tight with stitching issue.", confidence=.8,
            needs_human_review=False,
        )
