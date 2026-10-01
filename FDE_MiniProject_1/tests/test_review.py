from src.models import ClassificationResult
from src.review import apply_human_review


def test_human_review_updates_status_and_source():
    original = ClassificationResult(return_id="R1", sku_id="S1", category="Top", return_comment="not good", primary_reason="UNCERTAIN", normalized_summary="Unknown.", confidence=.2, needs_human_review=True, model_name="model", processing_status="HUMAN_REVIEW")
    updated = apply_human_review(original, "QUALITY", "QUALITY_OTHER", None)
    assert updated.processing_status.value == "ACCEPTED"
    assert updated.review_source.value == "HUMAN"

