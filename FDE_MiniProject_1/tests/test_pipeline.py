from dataclasses import replace

from src.config import Settings
from src.cost_tracker import UsageTracker
from src.models import ClassificationResult, EvaluationResult, ReturnRecord
from src.llm_client import LLMError, LLMResponseError, LLMUnavailableError
from src.pipeline import ReturnsPipeline


class FakeClassifier:
    def __init__(self, confidence=.9, reason="FIT", sub="TOO_TIGHT", review=False, name="fake"):
        self.confidence, self.reason, self.sub, self.review, self.name = confidence, reason, sub, review, name
        self.batch_calls = 0

    def classify(self, record, previous=None):
        return ClassificationResult(return_id=record.return_id, sku_id=record.sku_id, category=record.category, return_comment=record.return_comment, primary_reason=self.reason, sub_reason=self.sub, normalized_summary="Synthetic test classification.", confidence=self.confidence, needs_human_review=self.review, model_name=self.name, processing_status="HUMAN_REVIEW" if self.review else "ACCEPTED")

    def classify_batch(self, records):
        self.batch_calls += 1
        return [self.classify(item) for item in records]


class CountingClassifier(FakeClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def classify(self, record, previous=None):
        self.calls += 1
        return super().classify(record, previous)


class FakeEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, record, proposed):
        self.calls += 1
        return EvaluationResult(supported_by_comment=True, taxonomy_consistent=True, hallucination_detected=False, recommended_action="ACCEPT", explanation="Supported")

    def evaluate_batch(self, candidates):
        self.calls += 1
        return {
            record.return_id: EvaluationResult(
                supported_by_comment=True, taxonomy_consistent=True,
                hallucination_detected=False, recommended_action="ACCEPT", explanation="Supported",
            )
            for record, _ in candidates
        }


class RejectThenAcceptEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, record, proposed):
        self.calls += 1
        if self.calls == 1:
            return EvaluationResult(
                supported_by_comment=False, taxonomy_consistent=True,
                hallucination_detected=False, recommended_action="HUMAN_REVIEW",
                explanation="Bulk classification is not sufficiently supported.",
            )
        return EvaluationResult(
            supported_by_comment=True, taxonomy_consistent=True,
            hallucination_detected=False, recommended_action="ACCEPT",
            explanation="Strong classification is supported.",
        )

    def evaluate_batch(self, candidates):
        self.calls += 1
        accepted = self.calls > 1
        return {
            record.return_id: EvaluationResult(
                supported_by_comment=accepted,
                taxonomy_consistent=True,
                hallucination_detected=False,
                recommended_action="ACCEPT" if accepted else "HUMAN_REVIEW",
                explanation="Strong classification is supported." if accepted else "Bulk classification is not sufficiently supported.",
            )
            for record, _ in candidates
        }


class FailedEvaluator:
    def evaluate(self, record, proposed):
        raise LLMUnavailableError("provider down")

    def evaluate_batch(self, candidates):
        raise LLMUnavailableError("provider down")


class InvalidClassifier:
    def __init__(self):
        self.calls = 0

    def classify(self, record, previous=None):
        self.calls += 1
        raise LLMResponseError("invalid structured output")

    def classify_batch(self, records):
        self.calls += 1
        raise LLMResponseError("invalid structured output")


class FatalClassifier:
    def __init__(self, error=None):
        self.calls = 0
        self.error = error or LLMError("credential rejected", status_code=401, stage="bulk", model="test")

    def classify(self, record, previous=None):
        self.calls += 1
        raise self.error

    def classify_batch(self, records):
        self.calls += 1
        raise self.error


class ExplodingEvaluator:
    def evaluate(self, record, proposed):
        raise AssertionError("Evaluator should not run for an explicitly uncertain strong result")

    def evaluate_batch(self, candidates):
        raise AssertionError("Evaluator should not run for an explicitly uncertain strong result")


def record(comment="tight"):
    return ReturnRecord(return_id="R1", sku_id="S1", category="Top", return_reason="Other", return_comment=comment)


def test_blank_comment_skips_models():
    pipeline = ReturnsPipeline(FakeClassifier(), FakeClassifier(), FakeEvaluator(), UsageTracker(), Settings())
    output = pipeline.run([record("")])
    assert output.results[0].processing_status.value == "HUMAN_REVIEW"
    assert output.results[0].model_name == "none"


def test_easy_case_is_accepted():
    evaluator = FakeEvaluator()
    pipeline = ReturnsPipeline(FakeClassifier(), FakeClassifier(name="strong"), evaluator, UsageTracker(), Settings())
    output = pipeline.run([record()])
    assert output.results[0].processing_status.value == "ACCEPTED"
    assert output.results[0].model_name == "fake"
    assert evaluator.calls == 1


def test_evaluator_rejection_escalates_bulk_result_to_strong_and_evaluates_again():
    evaluator = RejectThenAcceptEvaluator()
    strong = CountingClassifier(name="strong")
    pipeline = ReturnsPipeline(FakeClassifier(), strong, evaluator, UsageTracker(), Settings())

    output = pipeline.run([record()])

    assert strong.calls == 1
    assert evaluator.calls == 2
    assert output.results[0].processing_status.value == "ACCEPTED"
    assert output.results[0].model_name == "strong"


def test_low_confidence_uses_strong_and_evaluator():
    pipeline = ReturnsPipeline(FakeClassifier(confidence=.2), FakeClassifier(name="strong"), FakeEvaluator(), UsageTracker(), Settings())
    output = pipeline.run([record()])
    assert output.results[0].model_name == "strong"


def test_evaluator_failure_is_visible_human_review():
    pipeline = ReturnsPipeline(FakeClassifier(confidence=.2), FakeClassifier(name="strong"), FailedEvaluator(), UsageTracker(), Settings())
    output = pipeline.run([record()])
    assert output.results[0].processing_status.value == "HUMAN_REVIEW"
    assert "Evaluator could not verify" in output.results[0].failure_reason


def test_structured_reason_bypasses_ai():
    bulk = CountingClassifier()
    pipeline = ReturnsPipeline(bulk, CountingClassifier(name="strong"), FakeEvaluator(), UsageTracker(), Settings())
    structured = ReturnRecord(return_id="R2", sku_id="S1", category="Top", return_reason="Not Delivered", return_comment="")
    output = pipeline.run([structured])
    assert bulk.calls == 0
    assert output.results[0].sub_reason.value == "NOT_DELIVERED"
    assert output.results[0].model_name == "structured-input"


def test_two_invalid_bulk_outputs_escalate_to_strong_model():
    bulk = InvalidClassifier()
    strong = FakeClassifier(name="strong")
    pipeline = ReturnsPipeline(bulk, strong, FakeEvaluator(), UsageTracker(), Settings())
    output = pipeline.run([record()])
    assert bulk.calls == 2
    assert output.results[0].processing_status.value == "ACCEPTED"
    assert output.results[0].model_name == "strong"


def test_uncertain_strong_result_goes_directly_to_human_review():
    pipeline = ReturnsPipeline(
        FakeClassifier(confidence=.2),
        FakeClassifier(reason="UNCERTAIN", sub=None, review=True, name="strong"),
        ExplodingEvaluator(), UsageTracker(), Settings(),
    )
    output = pipeline.run([record()])
    assert output.results[0].processing_status.value == "HUMAN_REVIEW"
    assert "insufficient evidence" in output.results[0].failure_reason


def test_fatal_provider_error_stops_new_ai_calls_but_keeps_all_records():
    bulk = FatalClassifier()
    settings = replace(Settings(), max_concurrent_requests=1, max_batch_size=2)
    records = [
        ReturnRecord(
            return_id=f"R{index}", sku_id="S1", category="Top",
            return_reason="Other", return_comment="tight",
        )
        for index in range(4)
    ]
    pipeline = ReturnsPipeline(bulk, FakeClassifier(name="strong"), FakeEvaluator(), UsageTracker(), settings)

    output = pipeline.run(records)

    assert bulk.calls == 1
    assert len(output.results) == 4
    assert all(item.processing_status.value == "FAILED" for item in output.results)
    assert all("AI processing stopped" in item.failure_reason for item in output.results)


def test_exhausted_provider_retries_open_pipeline_circuit():
    bulk = FatalClassifier(LLMUnavailableError("provider unavailable", stage="bulk", model="test"))
    settings = replace(Settings(), max_concurrent_requests=1, max_batch_size=10)
    records = [record(), record("loose")]
    records[1] = records[1].model_copy(update={"return_id": "R2"})
    pipeline = ReturnsPipeline(bulk, FakeClassifier(name="strong"), FakeEvaluator(), UsageTracker(), settings)

    output = pipeline.run(records)

    assert bulk.calls == 1
    assert len(output.results) == 2
    assert all(item.processing_status.value == "FAILED" for item in output.results)


def test_bulk_and_evaluator_use_configured_batches_of_fifty():
    bulk = FakeClassifier()
    strong = CountingClassifier(name="strong")
    evaluator = FakeEvaluator()
    settings = replace(Settings(), max_batch_size=50, evaluator_batch_size=50)
    records = [
        ReturnRecord(
            return_id=f"R{index:03d}", sku_id="S1", category="Top",
            return_reason="Other", return_comment="tight",
        )
        for index in range(101)
    ]
    pipeline = ReturnsPipeline(bulk, strong, evaluator, UsageTracker(), settings)

    output = pipeline.run(records)

    assert len(output.results) == 101
    assert bulk.batch_calls == 3
    assert evaluator.calls == 3
    assert strong.calls == 0
    assert all(item.processing_status.value == "ACCEPTED" for item in output.results)
