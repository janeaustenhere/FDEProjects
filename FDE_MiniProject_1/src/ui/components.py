from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..aggregation import (
    category_distribution,
    discovered_other_reasons,
    human_review_records,
    other_classification_metrics,
    other_classifications,
    sku_vendor_reason_distribution,
)


def render_reason_distribution(frame: pd.DataFrame, title: str = "Return reasons percentage") -> None:
    distribution = category_distribution(frame)
    if distribution.empty:
        st.warning("No accepted classifications match this view.")
        return
    chart, table = st.columns([1.35, 1])
    figure = px.pie(
        distribution,
        names="return_category",
        values="count",
        hole=0.35,
        title=title,
    )
    figure.update_traces(textposition="inside", textinfo="percent+label")
    chart.plotly_chart(figure, width="stretch")
    display = distribution.copy()
    display["percentage"] = display["percentage"].map(lambda value: f"{value:.1f}%")
    table.markdown("#### Return category percentages")
    table.dataframe(display, width="stretch", hide_index=True)


def render_sku_vendor_breakdown(frame: pd.DataFrame) -> None:
    st.markdown("#### SKU, vendor, and return-reason classification")
    st.caption("Percentage is calculated within each SKU/vendor group, using accepted classifications.")
    breakdown = sku_vendor_reason_distribution(frame)
    if breakdown.empty:
        st.info("No accepted classifications match this view.")
        return
    display = breakdown.rename(columns={
        "sku_id": "SKU",
        "vendor": "Vendor",
        "return_category": "Return reason",
        "count": "Returns",
        "percentage": "Reason percentage",
    })
    display["Reason percentage"] = display["Reason percentage"].map(lambda value: f"{value:.1f}%")
    st.dataframe(display, width="stretch", hide_index=True)


def render_other_analysis(frame: pd.DataFrame) -> None:
    st.markdown("#### AI classification of ‘Other’ returns")
    metrics = other_classification_metrics(frame)
    cards = st.columns(5)
    cards[0].metric("Other returns", metrics["total_other"])
    cards[1].metric("AI classified", metrics["ai_classified"])
    cards[2].metric("AI classification coverage", f"{metrics['coverage']:.1f}%")
    cards[3].metric("Human review", metrics["human_review"])
    cards[4].metric("Failed", metrics["failed"])
    st.caption("Coverage measures AI-accepted Other records, not accuracy. Accuracy requires human-labelled ground truth.")

    classified = other_classifications(frame)
    if classified.empty:
        st.info("No AI-accepted Other classifications match this view.")
    else:
        st.dataframe(classified, width="stretch", hide_index=True)

    discovered = discovered_other_reasons(frame)
    if not discovered.empty:
        st.markdown("#### Newly discovered reasons")
        st.caption("These comments contained a clear reason outside the existing controlled taxonomy.")
        st.dataframe(discovered, width="stretch", hide_index=True)

    review_queue = human_review_records(frame)
    st.markdown("#### Human review required")
    if review_queue.empty:
        st.success("No unresolved Other records match this view.")
    else:
        st.dataframe(review_queue, width="stretch", hide_index=True)
