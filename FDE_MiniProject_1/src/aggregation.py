from __future__ import annotations

import pandas as pd

from .models import ClassificationResult, ProcessingStatus, ReviewSource
from .reason_registry import (
    CLASSIFICATION_SOURCE_AI,
    CLASSIFICATION_SOURCE_STRUCTURED,
    REASON_MATCH_EXISTING,
    REASON_MATCH_NEW,
    business_category_for,
    is_other_reason,
)


def _label(value: str) -> str:
    return value.replace("_", " ").strip().title()


def results_frame(results: list[ClassificationResult]) -> pd.DataFrame:
    rows = []
    for item in results:
        row = item.model_dump(mode="json")
        row["secondary_reasons"] = ", ".join(
            f"{reason['primary_reason']}:{reason.get('sub_reason') or ''}" for reason in row["secondary_reasons"]
        )
        if is_other_reason(item.source_return_reason):
            if item.primary_reason.value == "OTHER_KNOWN":
                row["return_category"] = business_category_for(
                    item.primary_reason, item.sub_reason, item.discovered_category
                )
                row["return_reason_detail"] = item.discovered_reason or item.normalized_summary
            else:
                row["return_category"] = business_category_for(item.primary_reason, item.sub_reason)
                row["return_reason_detail"] = _label(item.sub_reason.value) if item.sub_reason else item.normalized_summary
            row["classification_source"] = CLASSIFICATION_SOURCE_AI
        else:
            row["return_category"] = item.source_return_reason.strip()
            row["return_reason_detail"] = item.source_return_reason.strip()
            row["classification_source"] = CLASSIFICATION_SOURCE_STRUCTURED
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        structured = frame.loc[frame["classification_source"] == CLASSIFICATION_SOURCE_STRUCTURED, "return_category"]
        existing_categories = set(structured.dropna().astype(str))
        frame["reason_match"] = REASON_MATCH_EXISTING
        ai_mask = frame["classification_source"] == CLASSIFICATION_SOURCE_AI
        frame.loc[ai_mask & ~frame["return_category"].isin(existing_categories), "reason_match"] = REASON_MATCH_NEW
    return frame


def accepted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return frame[frame["processing_status"] == ProcessingStatus.ACCEPTED.value].copy()


def reason_counts(frame: pd.DataFrame) -> pd.DataFrame:
    accepted = accepted_frame(frame)
    if accepted.empty:
        return pd.DataFrame(columns=["primary_reason", "count", "percentage"])
    counts = accepted.groupby("primary_reason", as_index=False).size().rename(columns={"size": "count"})
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100
    return counts.sort_values("count", ascending=False)


def category_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Unified distribution: structured selections plus AI-resolved Other rows."""
    accepted = accepted_frame(frame)
    if accepted.empty:
        return pd.DataFrame(columns=["return_category", "count", "percentage"])
    counts = accepted.groupby("return_category", as_index=False).size().rename(columns={"size": "count"})
    counts["percentage"] = counts["count"] / counts["count"].sum() * 100
    return counts.sort_values(["count", "return_category"], ascending=[False, True])


def sku_vendor_reason_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Return-reason mix for each SKU/vendor pair, with within-group percentages."""
    accepted = accepted_frame(frame)
    columns = ["sku_id", "vendor", "return_category", "count", "percentage"]
    if accepted.empty:
        return pd.DataFrame(columns=columns)
    grouped = (
        accepted.groupby(["sku_id", "vendor", "return_category"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    totals = grouped.groupby(["sku_id", "vendor"])["count"].transform("sum")
    grouped["percentage"] = grouped["count"] / totals * 100
    return grouped.sort_values(["sku_id", "count", "return_category"], ascending=[True, False, True])


def _other_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or "source_return_reason" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["source_return_reason"].fillna("").astype(str).map(is_other_reason)


def other_classification_metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    other = frame[_other_mask(frame)]
    total = len(other)
    ai_accepted = int(((other["processing_status"] == ProcessingStatus.ACCEPTED.value) & (other["review_source"] == ReviewSource.MODEL.value)).sum()) if total else 0
    human_review = int((other["processing_status"] == ProcessingStatus.HUMAN_REVIEW.value).sum()) if total else 0
    failed = int((other["processing_status"] == ProcessingStatus.FAILED.value).sum()) if total else 0
    return {
        "total_other": total,
        "ai_classified": ai_accepted,
        "coverage": ai_accepted / total * 100 if total else 0.0,
        "human_review": human_review,
        "failed": failed,
    }


def other_classifications(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["return_id", "sku_id", "vendor", "return_comment", "return_category", "return_reason_detail", "reason_match", "confidence"]
    if frame.empty:
        return pd.DataFrame(columns=["Return ID", "SKU ID", "Vendor", "Customer comment", "AI category", "AI reason", "Match type", "Confidence"])
    selected = frame[
        _other_mask(frame)
        & (frame["processing_status"] == ProcessingStatus.ACCEPTED.value)
        & (frame["review_source"] == ReviewSource.MODEL.value)
    ][columns].copy()
    selected["confidence"] = selected["confidence"].map(lambda value: f"{float(value):.1%}")
    return selected.rename(columns={
        "return_id": "Return ID", "sku_id": "SKU ID", "vendor": "Vendor",
        "return_comment": "Customer comment", "return_category": "AI category",
        "return_reason_detail": "AI reason", "reason_match": "Match type", "confidence": "Confidence",
    })


def discovered_other_reasons(frame: pd.DataFrame) -> pd.DataFrame:
    output_columns = ["Return ID", "SKU ID", "Vendor", "Customer comment", "New category", "Reason description", "Confidence"]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)
    selected = frame[
        _other_mask(frame)
        & (frame["processing_status"] == ProcessingStatus.ACCEPTED.value)
        & (frame["reason_match"] == REASON_MATCH_NEW)
    ][["return_id", "sku_id", "vendor", "return_comment", "return_category", "return_reason_detail", "confidence"]].copy()
    selected["confidence"] = selected["confidence"].map(lambda value: f"{float(value):.1%}")
    return selected.rename(columns={
        "return_id": "Return ID", "sku_id": "SKU ID", "vendor": "Vendor",
        "return_comment": "Customer comment", "return_category": "New category",
        "return_reason_detail": "Reason description", "confidence": "Confidence",
    })


def human_review_records(frame: pd.DataFrame) -> pd.DataFrame:
    output_columns = ["Return ID", "SKU ID", "Vendor", "Customer comment", "Suggested category", "Confidence", "Why human review is needed"]
    if frame.empty:
        return pd.DataFrame(columns=output_columns)
    selected = frame[
        _other_mask(frame) & (frame["processing_status"] == ProcessingStatus.HUMAN_REVIEW.value)
    ][["return_id", "sku_id", "vendor", "return_comment", "return_category", "confidence", "failure_reason", "normalized_summary"]].copy()
    selected["confidence"] = selected["confidence"].map(lambda value: f"{float(value):.1%}")
    selected["review_reason"] = selected["failure_reason"].fillna("").astype(str).str.strip()
    missing = selected["review_reason"].eq("")
    selected.loc[missing, "review_reason"] = selected.loc[missing, "normalized_summary"]
    return selected.rename(columns={
        "return_id": "Return ID", "sku_id": "SKU ID", "vendor": "Vendor",
        "return_comment": "Customer comment", "return_category": "Suggested category",
        "confidence": "Confidence", "review_reason": "Why human review is needed",
    })[output_columns]


def fit_breakdown(frame: pd.DataFrame) -> pd.DataFrame:
    accepted = accepted_frame(frame)
    fit = accepted[accepted["primary_reason"] == "FIT"] if not accepted.empty else accepted
    return fit.groupby("sub_reason", as_index=False).size().rename(columns={"size": "count"}).sort_values("count", ascending=False) if not fit.empty else pd.DataFrame(columns=["sub_reason", "count"])


def top_problem_skus(frame: pd.DataFrame) -> pd.DataFrame:
    accepted = accepted_frame(frame)
    if accepted.empty:
        return pd.DataFrame(columns=["sku_id", "vendor", "top_issue", "count"])
    grouped = accepted.groupby(["sku_id", "vendor", "return_category"], dropna=False).size().reset_index(name="count")
    grouped = grouped.sort_values(["sku_id", "vendor", "count"], ascending=[True, True, False]).drop_duplicates(["sku_id", "vendor"])
    return grouped.rename(columns={"return_category": "top_issue"}).sort_values("count", ascending=False)


def summary_metrics(frame: pd.DataFrame, total_cost: float = 0.0) -> dict[str, float | int | str]:
    total = len(frame)
    accepted = int((frame["processing_status"] == ProcessingStatus.ACCEPTED.value).sum()) if total else 0
    review = int((frame["processing_status"] == ProcessingStatus.HUMAN_REVIEW.value).sum()) if total else 0
    failed = int((frame["processing_status"] == ProcessingStatus.FAILED.value).sum()) if total else 0
    counts = category_distribution(frame)
    return {
        "total": total,
        "accepted": accepted,
        "review": review,
        "failed": failed,
        "coverage": accepted / total * 100 if total else 0,
        "review_rate": review / total * 100 if total else 0,
        "top_driver": counts.iloc[0]["return_category"] if not counts.empty else "—",
        "cost": total_cost,
    }
