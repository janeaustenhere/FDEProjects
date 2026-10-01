from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .reason_registry import is_other_reason


@dataclass(frozen=True)
class InputMetrics:
    overall_return_rate: float | None
    other_reason_rate: float | None
    return_rate_basis: str


def _normalise(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.casefold()


def derive_input_metrics(frame: pd.DataFrame) -> InputMetrics:
    """Calculate source-data KPIs without assuming an unavailable denominator."""
    other_rate: float | None = None
    if "return_reason" in frame.columns:
        reasons = _normalise(frame["return_reason"])
        stated = reasons.ne("")
        if stated.any():
            other_rate = float(reasons[stated].map(is_other_reason).mean() * 100)

    overall_rate: float | None = None
    basis = "Requires total_orders, is_returned, or order_status in the uploaded data."

    # Return-only extracts can provide their all-order denominator as metadata.
    if "total_orders" in frame.columns:
        totals = pd.to_numeric(frame["total_orders"], errors="coerce").dropna()
        positive = totals[totals > 0]
        if not positive.empty:
            total_orders = float(positive.iloc[0])
            return_count = int(frame["return_id"].fillna("").astype(str).str.strip().ne("").sum()) if "return_id" in frame.columns else len(frame)
            overall_rate = return_count / total_orders * 100
            basis = f"{return_count:,} return records ÷ {total_orders:,.0f} total orders"
    elif "is_returned" in frame.columns:
        values = _normalise(frame["is_returned"])
        recognised = values.isin({"true", "false", "1", "0", "yes", "no", "returned", "not_returned"})
        if recognised.any():
            returned = values[recognised].isin({"true", "1", "yes", "returned"})
            overall_rate = float(returned.mean() * 100)
            basis = f"is_returned across {int(recognised.sum()):,} recognised rows"
    elif "order_status" in frame.columns:
        statuses = _normalise(frame["order_status"])
        stated = statuses.ne("")
        if stated.any():
            returned = statuses[stated].isin({"return", "returned", "refunded"})
            overall_rate = float(returned.mean() * 100)
            basis = f"order_status across {int(stated.sum()):,} stated rows"

    return InputMetrics(overall_return_rate=overall_rate, other_reason_rate=other_rate, return_rate_basis=basis)


def format_rate(value: float | None) -> str:
    return "Not available" if value is None else f"{value:.1f}%"
