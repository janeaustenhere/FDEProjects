from src.classifier import Classifier
from src.config import Settings
from src.evaluator import Evaluator
from src.models import (
    BatchClassificationItem,
    BatchClassificationPayload,
    BatchEvaluationItem,
    BatchEvaluationPayload,
    ClassificationPayload,
    ClassificationResult,
    EvaluationResult,
    ReturnRecord,
)


class CapturingClient:
    def __init__(self):
        self.calls = []

    def structured_completion(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["response_model"] is ClassificationPayload:
            return ClassificationPayload(
                primary_reason="FIT", sub_reason="TOO_TIGHT", normalized_summary="Too tight.",
                confidence=.9, needs_human_review=False,
            )
        if kwargs["response_model"] is BatchClassificationPayload:
            return BatchClassificationPayload(classifications=[
                BatchClassificationItem(
                    return_id=item["return_id"], primary_reason="FIT", sub_reason="TOO_TIGHT",
                    normalized_summary="Too tight.", confidence=.9, needs_human_review=False,
                )
                for item in kwargs["user_payload"]["records"]
            ])
        if kwargs["response_model"] is BatchEvaluationPayload:
            return BatchEvaluationPayload(evaluations=[
                BatchEvaluationItem(
                    return_id=item["return_id"], supported_by_comment=True,
                    taxonomy_consistent=True, hallucination_detected=False,
                    recommended_action="ACCEPT", explanation="Supported.",
                )
                for item in kwargs["user_payload"]["candidates"]
            ])
        return EvaluationResult(
            supported_by_comment=True, taxonomy_consistent=True,
            hallucination_detected=False, recommended_action="ACCEPT", explanation="Supported.",
        )


def test_classifier_supplies_full_schema_and_sanitized_previous_result():
    client = CapturingClient()
    classifier = Classifier(client, Settings(), "strong")
    record = ReturnRecord(return_id="R1", sku_id="S1", category="Top", return_reason="Other", return_comment="tight")
    previous = ClassificationResult(
        return_id="R1", sku_id="S1", category="Top", return_comment="tight",
        primary_reason="FIT", sub_reason="TOO_TIGHT", normalized_summary="Too tight.",
        confidence=.8, needs_human_review=False, model_name="bulk", processing_status="ACCEPTED",
    )
    classifier.classify(record, previous)
    payload = client.calls[0]["user_payload"]
    assert "output_schema" in payload
    assert set(payload["previous_result"]) == set(ClassificationPayload.model_fields)
    assert "return_id" not in payload["previous_result"]


def test_evaluator_receives_only_classification_fields_and_output_schema():
    client = CapturingClient()
    evaluator = Evaluator(client, Settings())
    record = ReturnRecord(return_id="R1", sku_id="S1", category="Top", return_reason="Other", return_comment="tight")
    proposed = ClassificationResult(
        return_id="R1", sku_id="S1", category="Top", return_comment="tight",
        primary_reason="FIT", sub_reason="TOO_TIGHT", normalized_summary="Too tight.",
        confidence=.9, needs_human_review=False, model_name="strong", processing_status="ACCEPTED",
    )
    evaluator.evaluate(record, proposed)
    payload = client.calls[0]["user_payload"]
    assert set(payload["proposed_classification"]) == set(ClassificationPayload.model_fields)
    assert "output_schema" in payload


def test_bulk_classifier_sends_multiple_records_in_one_structured_call():
    client = CapturingClient()
    classifier = Classifier(client, Settings(), "bulk")
    records = [
        ReturnRecord(return_id="R1", sku_id="S1", category="Top", return_reason="Other", return_comment="tight"),
        ReturnRecord(return_id="R2", sku_id="S2", category="Dress", return_reason="Other", return_comment="loose"),
    ]

    results = classifier.classify_batch(records)

    assert len(client.calls) == 1
    assert client.calls[0]["response_model"] is BatchClassificationPayload
    assert [item["return_id"] for item in client.calls[0]["user_payload"]["records"]] == ["R1", "R2"]
    assert [item.return_id for item in results] == ["R1", "R2"]


def test_evaluator_sends_multiple_candidates_in_one_structured_call():
    client = CapturingClient()
    evaluator = Evaluator(client, Settings())
    pairs = []
    for return_id in ["R1", "R2"]:
        record = ReturnRecord(return_id=return_id, sku_id="S1", category="Top", return_reason="Other", return_comment="tight")
        proposed = ClassificationResult(
            return_id=return_id, sku_id="S1", category="Top", return_comment="tight",
            primary_reason="FIT", sub_reason="TOO_TIGHT", normalized_summary="Too tight.",
            confidence=.9, needs_human_review=False, model_name="bulk", processing_status="ACCEPTED",
        )
        pairs.append((record, proposed))

    results = evaluator.evaluate_batch(pairs)

    assert len(client.calls) == 1
    assert client.calls[0]["response_model"] is BatchEvaluationPayload
    assert set(results) == {"R1", "R2"}
