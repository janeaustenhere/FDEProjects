from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Event, Lock
from typing import Callable, Protocol

from .config import Settings
from .cost_tracker import UsageTracker
from .llm_client import LLMError, LLMResponseError, LLMUnavailableError
from .models import (
    ClassificationResult,
    EvaluationResult,
    PipelineResult,
    ProcessingStatus,
    RecommendedAction,
    ReturnRecord,
)
from .router import should_route
from .reason_registry import is_other_reason
from .source_reasons import structured_reason_result
from .taxonomy import PrimaryReason

logger = logging.getLogger(__name__)


class ClassifierLike(Protocol):
    def classify(self, record: ReturnRecord, previous: ClassificationResult | None = None) -> ClassificationResult: ...

    def classify_batch(self, records: list[ReturnRecord]) -> list[ClassificationResult]: ...


class EvaluatorLike(Protocol):
    def evaluate(self, record: ReturnRecord, proposed: ClassificationResult) -> EvaluationResult: ...

    def evaluate_batch(
        self, candidates: list[tuple[ReturnRecord, ClassificationResult]]
    ) -> dict[str, EvaluationResult]: ...


def blank_result(record: ReturnRecord) -> ClassificationResult:
    return ClassificationResult(
        return_id=record.return_id, sku_id=record.sku_id, category=record.category,
        return_comment=record.return_comment, source_return_reason=record.return_reason, vendor=record.vendor, primary_reason=PrimaryReason.UNCERTAIN,
        normalized_summary="No return comment was provided.", confidence=0,
        needs_human_review=True, model_name="none", processing_status=ProcessingStatus.HUMAN_REVIEW,
        failure_reason="No return comment provided.",
    )


def failed_result(record: ReturnRecord, reason: str) -> ClassificationResult:
    return ClassificationResult(
        return_id=record.return_id, sku_id=record.sku_id, category=record.category,
        return_comment=record.return_comment, source_return_reason=record.return_reason, vendor=record.vendor, primary_reason=PrimaryReason.UNCERTAIN,
        normalized_summary="The record could not be classified.", confidence=0,
        needs_human_review=True, model_name="unavailable", processing_status=ProcessingStatus.FAILED,
        failure_reason=reason,
    )


class ReturnsPipeline:
    def __init__(self, bulk: ClassifierLike, strong: ClassifierLike, evaluator: EvaluatorLike, tracker: UsageTracker, settings: Settings) -> None:
        self.bulk, self.strong, self.evaluator = bulk, strong, evaluator
        self.tracker, self.settings = tracker, settings
        self._fatal_event = Event()
        self._fatal_lock = Lock()
        self._fatal_reason: str | None = None

    @staticmethod
    def _is_run_fatal(exc: LLMError) -> bool:
        return isinstance(exc, LLMUnavailableError) or (
            not isinstance(exc, LLMResponseError) and exc.status_code in {400, 401, 403, 404, 422}
        )

    def _record_fatal_error(self, exc: LLMError) -> None:
        with self._fatal_lock:
            if not self._fatal_event.is_set():
                self._fatal_reason = str(exc)
                self._fatal_event.set()
                logger.error("Stopping new AI calls after fatal provider error: %s", exc)

    def _fatal_result(self, record: ReturnRecord) -> ClassificationResult:
        reason = self._fatal_reason or "Fatal AI provider configuration error"
        return failed_result(record, f"AI processing stopped for this run: {reason}")

    @staticmethod
    def _apply_evaluation(
        candidate: ClassificationResult,
        evaluation: EvaluationResult,
    ) -> tuple[bool, ClassificationResult]:
        accepted = (
            evaluation.recommended_action == RecommendedAction.ACCEPT
            and evaluation.supported_by_comment
            and evaluation.taxonomy_consistent
            and not evaluation.hallucination_detected
            and candidate.primary_reason != PrimaryReason.UNCERTAIN
            and not candidate.needs_human_review
        )
        if accepted:
            return True, candidate.model_copy(update={
                "processing_status": ProcessingStatus.ACCEPTED,
                "needs_human_review": False,
                "failure_reason": None,
            })
        return False, candidate.model_copy(update={
            "processing_status": ProcessingStatus.HUMAN_REVIEW,
            "needs_human_review": True,
            "failure_reason": evaluation.explanation,
        })

    def _strong_classify(
        self,
        record: ReturnRecord,
        previous: ClassificationResult | None,
    ) -> ClassificationResult:
        if self._fatal_event.is_set():
            return self._fatal_result(record)
        try:
            strong_result = self.strong.classify(record, previous)
            if strong_result.primary_reason == PrimaryReason.UNCERTAIN or strong_result.needs_human_review:
                return strong_result.model_copy(update={
                    "processing_status": ProcessingStatus.HUMAN_REVIEW,
                    "needs_human_review": True,
                    "failure_reason": "Strong classifier found insufficient evidence for safe automated acceptance.",
                })
            return strong_result
        except LLMResponseError as exc:
            return failed_result(record, f"Strong model validation failure: {exc}").model_copy(
                update={"processing_status": ProcessingStatus.HUMAN_REVIEW}
            )
        except LLMError as exc:
            if self._is_run_fatal(exc):
                self._record_fatal_error(exc)
                return self._fatal_result(record)
            return failed_result(record, str(exc))
        except Exception as exc:  # one record must not fail the whole job
            logger.exception("Unexpected record failure return_id=%s", record.return_id)
            return failed_result(record, f"Unexpected processing error: {exc}")

    def _bulk_batch(self, records: list[ReturnRecord]) -> list[ClassificationResult] | None:
        try:
            return self.bulk.classify_batch(records)
        except LLMResponseError as first_error:
            try:
                return self.bulk.classify_batch(records)
            except LLMResponseError:
                logger.warning(
                    "Bulk batch validation failed twice; escalating %s records error=%s",
                    len(records),
                    first_error,
                )
                return None

    def _evaluator_batch(
        self,
        candidates: list[tuple[ReturnRecord, ClassificationResult]],
    ) -> dict[str, EvaluationResult]:
        try:
            return self.evaluator.evaluate_batch(candidates)
        except LLMResponseError as first_error:
            try:
                return self.evaluator.evaluate_batch(candidates)
            except LLMResponseError as exc:
                raise LLMResponseError(
                    f"Evaluator batch validation failed twice: {first_error}",
                    stage=exc.stage,
                    model=exc.model,
                ) from exc

    def run(self, records: list[ReturnRecord], progress: Callable[[int, int], None] | None = None) -> PipelineResult:
        started = time.perf_counter()
        self.tracker.reset()
        self._fatal_event.clear()
        self._fatal_reason = None
        results: list[ClassificationResult | None] = [None] * len(records)
        completed_count = 0

        def finalize(index: int, result: ClassificationResult) -> None:
            nonlocal completed_count
            if results[index] is not None:
                return
            results[index] = result
            completed_count += 1
            if progress:
                progress(completed_count, len(records))

        ai_records: list[tuple[int, ReturnRecord]] = []
        for index, record in enumerate(records):
            if not is_other_reason(record.return_reason):
                finalize(index, structured_reason_result(record))
            elif not record.return_comment.strip():
                finalize(index, blank_result(record))
            else:
                ai_records.append((index, record))

        evaluation_candidates: list[tuple[int, ReturnRecord, ClassificationResult, str]] = []
        strong_tasks: list[tuple[int, ReturnRecord, ClassificationResult | None]] = []

        for start in range(0, len(ai_records), self.settings.max_batch_size):
            indexed_batch = ai_records[start : start + self.settings.max_batch_size]
            if self._fatal_event.is_set():
                for index, record in indexed_batch:
                    finalize(index, self._fatal_result(record))
                continue
            batch_records = [record for _, record in indexed_batch]
            try:
                bulk_results = self._bulk_batch(batch_records)
            except LLMError as exc:
                if self._is_run_fatal(exc):
                    self._record_fatal_error(exc)
                for index, record in indexed_batch:
                    finalize(index, self._fatal_result(record) if self._fatal_event.is_set() else failed_result(record, str(exc)))
                continue
            except Exception as exc:
                logger.exception("Unexpected bulk batch failure")
                for index, record in indexed_batch:
                    finalize(index, failed_result(record, f"Unexpected bulk processing error: {exc}"))
                continue

            if bulk_results is None:
                strong_tasks.extend((index, record, None) for index, record in indexed_batch)
                continue
            for (index, record), bulk_result in zip(indexed_batch, bulk_results, strict=True):
                route, route_reason = should_route(bulk_result, self.settings.confidence_threshold)
                logger.info("Route decision return_id=%s route=%s reason=%s", record.return_id, route, route_reason)
                if route:
                    strong_tasks.append((index, record, bulk_result))
                else:
                    evaluation_candidates.append((index, record, bulk_result, "bulk"))

        def run_strong(
            tasks: list[tuple[int, ReturnRecord, ClassificationResult | None]],
        ) -> list[tuple[int, ReturnRecord, ClassificationResult, str]]:
            candidates: list[tuple[int, ReturnRecord, ClassificationResult, str]] = []
            if not tasks:
                return candidates
            workers = max(1, min(self.settings.max_concurrent_requests, len(tasks)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._strong_classify, record, previous): (index, record)
                    for index, record, previous in tasks
                }
                for future in as_completed(futures):
                    index, record = futures[future]
                    result = future.result()
                    if result.processing_status == ProcessingStatus.ACCEPTED and not result.needs_human_review:
                        candidates.append((index, record, result, "strong"))
                    else:
                        finalize(index, result)
            return sorted(candidates, key=lambda item: item[0])

        evaluation_candidates.extend(run_strong(strong_tasks))
        evaluation_candidates.sort(key=lambda item: item[0])

        def evaluate_round(
            candidates: list[tuple[int, ReturnRecord, ClassificationResult, str]],
            *,
            escalate_bulk_rejections: bool,
        ) -> list[tuple[int, ReturnRecord, ClassificationResult | None]]:
            escalations: list[tuple[int, ReturnRecord, ClassificationResult | None]] = []
            size = self.settings.evaluator_batch_size
            for start in range(0, len(candidates), size):
                batch = candidates[start : start + size]
                if self._fatal_event.is_set():
                    reason = self._fatal_reason or "AI provider unavailable"
                    for index, _, candidate, _ in batch:
                        finalize(index, candidate.model_copy(update={
                            "processing_status": ProcessingStatus.HUMAN_REVIEW,
                            "needs_human_review": True,
                            "failure_reason": f"Evaluator could not verify classification: {reason}",
                        }))
                    continue
                try:
                    evaluations = self._evaluator_batch([
                        (record, candidate) for _, record, candidate, _ in batch
                    ])
                except LLMError as exc:
                    if self._is_run_fatal(exc):
                        self._record_fatal_error(exc)
                    for index, _, candidate, _ in batch:
                        finalize(index, candidate.model_copy(update={
                            "processing_status": ProcessingStatus.HUMAN_REVIEW,
                            "needs_human_review": True,
                            "failure_reason": f"Evaluator could not verify classification: {exc}",
                        }))
                    continue
                except Exception as exc:
                    logger.exception("Unexpected evaluator batch failure")
                    for index, _, candidate, _ in batch:
                        finalize(index, candidate.model_copy(update={
                            "processing_status": ProcessingStatus.HUMAN_REVIEW,
                            "needs_human_review": True,
                            "failure_reason": f"Unexpected evaluator error: {exc}",
                        }))
                    continue

                for index, record, candidate, source in batch:
                    accepted, evaluated = self._apply_evaluation(
                        candidate, evaluations[record.return_id]
                    )
                    if accepted:
                        finalize(index, evaluated)
                    elif source == "bulk" and escalate_bulk_rejections:
                        logger.info(
                            "Escalating evaluator-rejected bulk result return_id=%s reason=%s",
                            record.return_id,
                            evaluated.failure_reason,
                        )
                        escalations.append((index, record, candidate))
                    else:
                        finalize(index, evaluated)
            return escalations

        evaluator_escalations = evaluate_round(
            evaluation_candidates,
            escalate_bulk_rejections=True,
        )
        reevaluation_candidates = run_strong(evaluator_escalations)
        evaluate_round(reevaluation_candidates, escalate_bulk_rejections=False)

        for index, record in enumerate(records):
            if results[index] is None:
                finalize(index, failed_result(record, "Record did not reach a terminal pipeline state."))
        final_results = [item for item in results if item is not None]
        return PipelineResult(
            results=final_results,
            usage=self.tracker.summary(),
            processing_seconds=time.perf_counter() - started,
            metadata={
                "records": len(records),
                "confidence_threshold": self.settings.confidence_threshold,
                "bulk_batch_size": self.settings.max_batch_size,
                "evaluator_batch_size": self.settings.evaluator_batch_size,
            },
        )
