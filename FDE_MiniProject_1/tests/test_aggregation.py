from src.aggregation import (
    category_distribution,
    discovered_other_reasons,
    human_review_records,
    other_classification_metrics,
    other_classifications,
    reason_counts,
    results_frame,
    sku_vendor_reason_distribution,
    summary_metrics,
)
from src.models import ClassificationResult


def result(return_id, reason, sub, status="ACCEPTED"):
    return ClassificationResult(return_id=return_id, sku_id="S1", category="Top", return_comment="comment", primary_reason=reason, sub_reason=sub, normalized_summary="Summary.", confidence=.9, needs_human_review=status != "ACCEPTED", model_name="test", processing_status=status)


def test_aggregation_uses_accepted_primary_only():
    frame = results_frame([result("R1", "FIT", "TOO_TIGHT"), result("R2", "QUALITY", "STITCHING"), result("R3", "UNCERTAIN", None, "HUMAN_REVIEW")])
    counts = reason_counts(frame)
    assert counts["count"].sum() == 2
    assert round(counts["percentage"].sum()) == 100
    assert summary_metrics(frame)["coverage"] == pytest.approx(200 / 3)


def test_unified_distribution_combines_source_and_ai_categories():
    structured = result("R1", "DELIVERY_OR_PACKAGING", "NOT_DELIVERED").model_copy(
        update={"source_return_reason": "Not Delivered", "model_name": "structured-input"}
    )
    ai = result("R2", "FIT", "TOO_SMALL").model_copy(update={"source_return_reason": "Other"})
    distribution = category_distribution(results_frame([structured, ai]))
    assert set(distribution["return_category"]) == {"Not Delivered", "Size Issue"}
    assert distribution["percentage"].sum() == pytest.approx(100)


def test_new_ai_category_and_reason_are_preserved():
    discovered = ClassificationResult(
        return_id="R4", sku_id="S1", category="Top", return_comment="allergic reaction",
        source_return_reason="Other", primary_reason="OTHER_KNOWN",
        discovered_category="Skin Irritation", discovered_reason="Fabric caused itching.",
        normalized_summary="Customer reports skin irritation.", confidence=.9,
        needs_human_review=False, model_name="test", processing_status="ACCEPTED",
    )
    frame = results_frame([discovered])
    assert frame.iloc[0]["return_category"] == "Skin Irritation"
    assert frame.iloc[0]["return_reason_detail"] == "Fabric caused itching."
    discovered_table = discovered_other_reasons(frame)
    assert discovered_table.iloc[0]["New category"] == "Skin Irritation"
    assert discovered_table.iloc[0]["Reason description"] == "Fabric caused itching."


def test_sku_vendor_percentages_are_within_each_group():
    first = result("R1", "FIT", "TOO_SMALL").model_copy(update={"vendor": "Vendor A"})
    second = result("R2", "FIT", "TOO_SMALL").model_copy(update={"vendor": "Vendor A"})
    third = result("R3", "QUALITY", "STITCHING").model_copy(update={"vendor": "Vendor A"})
    grouped = sku_vendor_reason_distribution(results_frame([first, second, third]))
    percentages = dict(zip(grouped["return_category"], grouped["percentage"]))
    assert percentages["Size Issue"] == pytest.approx(200 / 3)
    assert percentages["Quality"] == pytest.approx(100 / 3)
    assert grouped["percentage"].sum() == pytest.approx(100)


def test_other_coverage_and_human_review_details():
    accepted = result("R1", "FIT", "TOO_SMALL")
    review = result("R2", "UNCERTAIN", None, "HUMAN_REVIEW").model_copy(
        update={"failure_reason": "Insufficient information to determine a reason."}
    )
    frame = results_frame([accepted, review])
    metrics = other_classification_metrics(frame)
    assert metrics == {"total_other": 2, "ai_classified": 1, "coverage": 50, "human_review": 1, "failed": 0}
    assert other_classifications(frame).iloc[0]["Return ID"] == "R1"
    queue = human_review_records(frame)
    assert queue.iloc[0]["Return ID"] == "R2"
    assert queue.iloc[0]["SKU ID"] == "S1"
    assert "Insufficient information" in queue.iloc[0]["Why human review is needed"]


import pytest
