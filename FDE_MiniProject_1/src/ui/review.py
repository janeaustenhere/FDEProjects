from __future__ import annotations

import streamlit as st

from ..models import ProcessingStatus, ReviewSource
from ..review import apply_human_review
from ..taxonomy import BodyArea, PrimaryReason, allowed_sub_reasons


def render() -> None:
    st.title("Human Review")
    pipeline = st.session_state.get("pipeline_result")
    if pipeline is None:
        st.info("Analyse a dataset first.")
        return
    queue = [(index, item) for index, item in enumerate(pipeline.results) if item.processing_status.value == "HUMAN_REVIEW"]
    if not queue:
        st.success("The human-review queue is clear.")
        return
    st.caption(f"{len(queue)} record(s) need a decision. Human corrections update the dashboard for this session; they do not retrain a model.")
    options = {f"{item.return_id} · {item.sku_id}": (index, item) for index, item in queue}
    selected = st.selectbox("Review record", options)
    index, item = options[selected]
    st.markdown(f"**Original comment:** {item.return_comment or '— (blank)'}")
    cols = st.columns(3)
    cols[0].metric("Suggested reason", item.primary_reason.value.replace("_", " ").title())
    cols[1].metric("Confidence", f"{item.confidence:.2f}")
    cols[2].metric("Category", item.category)
    st.warning(item.failure_reason or "The model requested human review.")
    primary = st.selectbox("Primary reason", [reason.value for reason in PrimaryReason if reason != PrimaryReason.UNCERTAIN])
    subs = allowed_sub_reasons(primary)
    sub = st.selectbox("Sub-reason", subs) if subs else None
    body = st.selectbox("Body area (optional)", [None] + [area.value for area in BodyArea]) if primary == "FIT" else None
    discovered_category = st.text_input("New category") if primary == "OTHER_KNOWN" else None
    discovered_reason = st.text_input("Interpreted reason") if primary == "OTHER_KNOWN" else None
    c1, c2 = st.columns(2)
    invalid_new_category = primary == "OTHER_KNOWN" and not (discovered_category and discovered_reason)
    if c1.button("Confirm classification", type="primary", width="stretch", disabled=invalid_new_category):
        pipeline.results[index] = apply_human_review(item, primary, sub, body, discovered_category, discovered_reason)
        st.success("Classification accepted and dashboard updated.")
        st.rerun()
    if c2.button("Mark unclassifiable", width="stretch"):
        pipeline.results[index] = item.model_copy(update={
            "processing_status": ProcessingStatus.FAILED,
            "review_source": ReviewSource.HUMAN,
            "failure_reason": "Human reviewer marked this record unclassifiable.",
        })
        st.info("Record marked unclassifiable and excluded from accepted aggregates.")
        st.rerun()
