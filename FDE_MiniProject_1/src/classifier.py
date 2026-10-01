from __future__ import annotations

from pathlib import Path

from .config import ROOT, Settings
from .llm_client import LLMClient
from .llm_client import LLMResponseError
from .models import (
    BatchClassificationPayload,
    ClassificationPayload,
    ClassificationResult,
    ProcessingStatus,
    ReturnRecord,
)
from .taxonomy import SUB_REASONS_BY_PRIMARY


class Classifier:
    def __init__(self, client: LLMClient, settings: Settings, stage: str) -> None:
        if stage not in {"bulk", "strong"}:
            raise ValueError("Classifier stage must be bulk or strong")
        self.client = client
        self.settings = settings
        self.stage = stage
        self.model_name = settings.bulk_model if stage == "bulk" else settings.strong_model
        prompt_name = "bulk_classifier.txt" if stage == "bulk" else "strong_classifier.txt"
        self.prompt = (ROOT / "prompts" / prompt_name).read_text(encoding="utf-8")

    @staticmethod
    def _taxonomy_payload() -> dict[str, list[str]]:
        return {
            primary.value: sorted(sub.value for sub in subs)
            for primary, subs in SUB_REASONS_BY_PRIMARY.items()
        }

    def _result(self, record: ReturnRecord, payload: ClassificationPayload) -> ClassificationResult:
        status = ProcessingStatus.HUMAN_REVIEW if payload.needs_human_review else ProcessingStatus.ACCEPTED
        return ClassificationResult(
            **payload.model_dump(exclude={"return_id"}),
            return_id=record.return_id,
            sku_id=record.sku_id,
            category=record.category,
            return_comment=record.return_comment,
            source_return_reason=record.return_reason,
            vendor=record.vendor,
            model_name=self.model_name,
            processing_status=status,
        )

    def classify(self, record: ReturnRecord, previous: ClassificationResult | None = None) -> ClassificationResult:
        output_schema = ClassificationPayload.model_json_schema()
        payload = {
            "customer_comment": record.return_comment,
            "category": record.category,
            "vendor": record.vendor,
            "allowed_taxonomy": self._taxonomy_payload(),
            "allowed_body_areas": ["CHEST", "WAIST", "HIP", "SHOULDER", "SLEEVE", "LENGTH", "OTHER"],
            "output_schema": output_schema,
        }
        if previous is not None:
            classification_fields = set(ClassificationPayload.model_fields)
            payload["previous_result"] = previous.model_dump(mode="json", include=classification_fields)
        result = self.client.structured_completion(
            stage=self.stage,
            model=self.model_name,
            system_prompt=self.prompt,
            user_payload=payload,
            response_model=ClassificationPayload,
            pricing=self.settings.pricing(self.stage),
        )
        return self._result(record, result)

    def classify_batch(self, records: list[ReturnRecord]) -> list[ClassificationResult]:
        if not records:
            return []
        if self.stage != "bulk":
            raise ValueError("Batch classification is only supported by the bulk classifier")
        response = self.client.structured_completion(
            stage=self.stage,
            model=self.model_name,
            system_prompt=self.prompt,
            user_payload={
                "records": [
                    {
                        "return_id": record.return_id,
                        "customer_comment": record.return_comment,
                        "category": record.category,
                        "vendor": record.vendor,
                    }
                    for record in records
                ],
                "allowed_taxonomy": self._taxonomy_payload(),
                "allowed_body_areas": ["CHEST", "WAIST", "HIP", "SHOULDER", "SLEEVE", "LENGTH", "OTHER"],
                "output_schema": BatchClassificationPayload.model_json_schema(),
            },
            response_model=BatchClassificationPayload,
            pricing=self.settings.pricing(self.stage),
        )
        requested = {record.return_id: record for record in records}
        returned = {item.return_id: item for item in response.classifications}
        if set(returned) != set(requested):
            missing = sorted(set(requested) - set(returned))
            unexpected = sorted(set(returned) - set(requested))
            raise LLMResponseError(
                f"Bulk response return_id mismatch; missing={missing}, unexpected={unexpected}",
                stage=self.stage,
                model=self.model_name,
            )
        return [self._result(record, returned[record.return_id]) for record in records]
