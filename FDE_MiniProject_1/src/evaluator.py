from __future__ import annotations

from .config import ROOT, Settings
from .llm_client import LLMClient, LLMResponseError
from .models import (
    BatchEvaluationPayload,
    ClassificationPayload,
    ClassificationResult,
    EvaluationResult,
    ReturnRecord,
)


class Evaluator:
    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings
        self.prompt = (ROOT / "prompts" / "evaluator.txt").read_text(encoding="utf-8")

    def evaluate(self, record: ReturnRecord, proposed: ClassificationResult) -> EvaluationResult:
        classification_fields = set(ClassificationPayload.model_fields)
        return self.client.structured_completion(
            stage="evaluator",
            model=self.settings.evaluator_model,
            system_prompt=self.prompt,
            user_payload={
                "original_customer_comment": record.return_comment,
                "proposed_classification": proposed.model_dump(mode="json", include=classification_fields),
                "output_schema": EvaluationResult.model_json_schema(),
            },
            response_model=EvaluationResult,
            pricing=self.settings.pricing("evaluator"),
        )

    def evaluate_batch(
        self,
        candidates: list[tuple[ReturnRecord, ClassificationResult]],
    ) -> dict[str, EvaluationResult]:
        if not candidates:
            return {}
        classification_fields = set(ClassificationPayload.model_fields)
        response = self.client.structured_completion(
            stage="evaluator",
            model=self.settings.evaluator_model,
            system_prompt=self.prompt,
            user_payload={
                "candidates": [
                    {
                        "return_id": record.return_id,
                        "original_customer_comment": record.return_comment,
                        "proposed_classification": proposed.model_dump(
                            mode="json", include=classification_fields
                        ),
                    }
                    for record, proposed in candidates
                ],
                "output_schema": BatchEvaluationPayload.model_json_schema(),
            },
            response_model=BatchEvaluationPayload,
            pricing=self.settings.pricing("evaluator"),
        )
        requested = {record.return_id for record, _ in candidates}
        returned = {item.return_id: item for item in response.evaluations}
        if set(returned) != requested:
            missing = sorted(requested - set(returned))
            unexpected = sorted(set(returned) - requested)
            raise LLMResponseError(
                f"Evaluator response return_id mismatch; missing={missing}, unexpected={unexpected}",
                stage="evaluator",
                model=self.settings.evaluator_model,
            )
        return {
            return_id: EvaluationResult.model_validate(item.model_dump(exclude={"return_id"}))
            for return_id, item in returned.items()
        }
