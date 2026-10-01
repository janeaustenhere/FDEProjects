from __future__ import annotations

import plotly.express as px
import streamlit as st

from ..aggregation import (
    fit_breakdown,
    results_frame,
    summary_metrics,
    top_problem_skus,
)
from ..input_metrics import format_rate
from .components import render_other_analysis, render_reason_distribution, render_sku_vendor_breakdown


def _multiselect(label: str, frame, column: str):
    options = sorted(value for value in frame[column].dropna().unique().tolist() if value)
    return st.multiselect(label, options)


def render() -> None:
    st.title("Returns Intelligence")
    pipeline = st.session_state.get("pipeline_result")
    if pipeline is None:
        st.info("Analyse a dataset first.")
        return
    frame = results_frame(pipeline.results)
    total_cost = sum(item.estimated_cost for item in pipeline.usage)
    source_metrics = st.session_state.get("input_metrics")
    if source_metrics is not None:
        st.subheader("Source-data metrics")
        source_cards = st.columns(2)
        source_cards[0].metric("Overall return rate", format_rate(source_metrics.overall_return_rate), help=source_metrics.return_rate_basis)
        source_cards[1].metric("Reasons captured as ‘Other’", format_rate(source_metrics.other_reason_rate), help="Calculated from non-blank return_reason values in the uploaded dataset.")
    with st.expander("Filters", expanded=True):
        cols = st.columns(6)
        selected = {
            "category": _multiselect("Category", frame, "category"),
            "sku_id": _multiselect("SKU", frame, "sku_id"),
            "vendor": _multiselect("Vendor", frame, "vendor"),
            "return_category": _multiselect("Return category", frame, "return_category"),
            "sub_reason": _multiselect("Sub-reason", frame, "sub_reason"),
            "processing_status": _multiselect("Status", frame, "processing_status"),
        }
    filtered = frame.copy()
    for column, values in selected.items():
        if values:
            filtered = filtered[filtered[column].isin(values)]
    metrics = summary_metrics(filtered, total_cost)
    cards = st.columns(5)
    cards[0].metric("Comments analysed", metrics["total"])
    cards[1].metric("Classification coverage", f"{metrics['coverage']:.1f}%")
    cards[2].metric("Human review rate", f"{metrics['review_rate']:.1f}%")
    cards[3].metric("Top return driver", str(metrics["top_driver"]).replace("_", " ").title())
    cards[4].metric("Model cost", f"${metrics['cost']:.4f}")

    render_reason_distribution(filtered, title="Return category distribution")

    left, right = st.columns(2)
    fit = fit_breakdown(filtered)
    if not fit.empty:
        left.plotly_chart(px.bar(fit, x="sub_reason", y="count", title="Fit breakdown"), width="stretch")
    else:
        left.info("No accepted fit classifications in this view.")
    accepted = filtered[filtered["processing_status"] == "ACCEPTED"]
    if not accepted.empty:
        category = accepted.groupby(["category", "return_category"]).size().reset_index(name="count")
        right.plotly_chart(px.bar(category, x="category", y="count", color="return_category", title="Return reason by product category"), width="stretch")
    st.subheader("Top problem SKUs")
    st.dataframe(top_problem_skus(filtered), width="stretch", hide_index=True)

    render_sku_vendor_breakdown(filtered)
    render_other_analysis(filtered)

    st.subheader("Evidence behind the aggregate")
    reasons = sorted(accepted["return_category"].unique()) if not accepted.empty else []
    reason = st.selectbox("Inspect representative comments", reasons, index=None, placeholder="Select a category")
    if reason:
        evidence = accepted[accepted["return_category"] == reason][["return_id", "sku_id", "vendor", "source_return_reason", "return_comment", "return_reason_detail", "classification_source"]].head(10)
        st.dataframe(evidence, width="stretch", hide_index=True)
    st.download_button("Download classified results", filtered.to_csv(index=False).encode(), "dhaga_classified_returns.csv", "text/csv")
