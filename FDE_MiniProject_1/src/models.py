from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .reason_registry import DEFAULT_VENDOR
from .taxonomy import BodyArea, PrimaryReason, SUB_REASONS_BY_PRIMARY, SubReason


class ProcessingStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    FAILED = "FAILED"


class ReviewSource(str, Enum):
    MODEL = "MODEL"
    HUMAN = "HUMAN"


class ReturnRecord(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    return_id: str
    sku_id: str
    category: str
    return_reason: str
    return_comment: str = ""
    vendor: str = DEFAULT_VENDOR


class SecondaryReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_reason: PrimaryReason
    sub_reason: SubReason | None = None

    @model_validator(mode="after")
    def validate_pair(self) -> "SecondaryReason":
        if self.primary_reason in {PrimaryReason.OTHER_KNOWN, PrimaryReason.UNCERTAIN}:
            raise ValueError("OTHER_KNOWN and UNCERTAIN cannot be secondary reasons")
        if self.sub_reason not in SUB_REASONS_BY_PRIMARY[self.primary_reason]:
            raise ValueError("Sub-reason is incompatible with primary reason")
        return self


class ClassificationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    primary_reason: PrimaryReason
    sub_reason: SubReason | None = None
    body_area: BodyArea | None = None
    secondary_reasons: list[SecondaryReason] = Field(default_factory=list)
    normalized_summary: str = Field(min_length=1)
    discovered_category: str | None = None
    discovered_reason: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool

    @model_validator(mode="after")
    def validate_taxonomy(self) -> "ClassificationPayload":
        allowed = SUB_REASONS_BY_PRIMARY[self.primary_reason]
        if allowed and self.sub_reason not in allowed:
            raise ValueError("A valid sub-reason is required for this primary reason")
        if not allowed and self.sub_reason is not None:
            raise ValueError("This primary reason does not accept a sub-reason")
        if self.body_area is not None and self.primary_reason != PrimaryReason.FIT:
            raise ValueError("Body area is only valid for FIT")
        if self.primary_reason == PrimaryReason.UNCERTAIN and not self.needs_human_review:
            raise ValueError("UNCERTAIN must require human review")
        if self.primary_reason == PrimaryReason.OTHER_KNOWN:
            if not (self.discovered_category and self.discovered_category.strip()):
                raise ValueError("OTHER_KNOWN requires discovered_category")
            if not (self.discovered_reason and self.discovered_reason.strip()):
                raise ValueError("OTHER_KNOWN requires discovered_reason")
        elif self.discovered_category is not None or self.discovered_reason is not None:
            raise ValueError("Discovered fields are only valid for OTHER_KNOWN")

        seen: set[tuple[PrimaryReason, SubReason | None]] = set()
        primary_pair = (self.primary_reason, self.sub_reason)
        for secondary in self.secondary_reasons:
            pair = (secondary.primary_reason, secondary.sub_reason)
            if pair == primary_pair:
                raise ValueError("Secondary reason duplicates the primary classification")
            if pair in seen:
                raise ValueError("Duplicate secondary reason")
            seen.add(pair)
        return self


class BatchClassificationItem(ClassificationPayload):
    return_id: str = Field(min_length=1)


class BatchClassificationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classifications: list[BatchClassificationItem] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_return_ids(self) -> "BatchClassificationPayload":
        ids = [item.return_id for item in self.classifications]
        if len(ids) != len(set(ids)):
            raise ValueError("Batch classification contains duplicate return_id values")
        return self


class ClassificationResult(ClassificationPayload):
    return_id: str
    sku_id: str
    category: str
    return_comment: str
    source_return_reason: str = "Other"
    vendor: str = DEFAULT_VENDOR
    model_name: str
    processing_status: ProcessingStatus
    failure_reason: str | None = None
    review_source: ReviewSource = ReviewSource.MODEL


class RecommendedAction(str, Enum):
    ACCEPT = "ACCEPT"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    supported_by_comment: bool
    taxonomy_consistent: bool
    hallucination_detected: bool
    recommended_action: RecommendedAction
    explanation: str = Field(min_length=1)


class BatchEvaluationItem(EvaluationResult):
    return_id: str = Field(min_length=1)


class BatchEvaluationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluations: list[BatchEvaluationItem] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_return_ids(self) -> "BatchEvaluationPayload":
        ids = [item.return_id for item in self.evaluations]
        if len(ids) != len(set(ids)):
            raise ValueError("Batch evaluation contains duplicate return_id values")
        return self


class ModelUsage(BaseModel):
    stage: str
    model_name: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    estimated_cost: float = 0.0


class ValidationIssue(BaseModel):
    row_number: int | None = None
    return_id: str | None = None
    issue: str


class ValidationReport(BaseModel):
    total_rows: int
    valid_rows: int
    blank_comments: int
    duplicate_ids: int
    issues: list[ValidationIssue] = Field(default_factory=list)


class PipelineResult(BaseModel):
    results: list[ClassificationResult]
    usage: list[ModelUsage]
    processing_seconds: float
    metadata: dict[str, Any] = Field(default_factory=dict)
