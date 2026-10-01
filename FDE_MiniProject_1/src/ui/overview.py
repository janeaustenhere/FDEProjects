from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..aggregation import results_frame
from ..config import Settings
from ..input_metrics import derive_input_metrics, format_rate
from ..reason_registry import DEFAULT_VENDOR
from ..services import build_pipeline
from .components import render_other_analysis, render_reason_distribution, render_sku_vendor_breakdown
from ..validators import InputValidationError, read_csv, validate_dataframe


def render(settings: Settings, demo_path: Path) -> None:
    st.title("Dhaga & Co.")
    st.subheader("Returns Intelligence")
    st.caption("Turn unstructured return feedback into actionable category intelligence.")
    st.info("Existing return reasons are counted directly. Only comments whose return reason is ‘Other’ are interpreted and categorised by AI.")

    uploaded = st.file_uploader("Upload returns CSV", type=["csv"])
    if st.button("Use demo dataset", width="stretch"):
        st.session_state.input_frame = read_csv(demo_path)
        st.session_state.input_source = "Synthetic demo dataset"
        st.session_state.input_key = "demo"
        st.session_state.pop("pipeline_result", None)
    if uploaded is not None:
        if uploaded.size > settings.max_upload_mb * 1024 * 1024:
            st.error(f"File exceeds the {settings.max_upload_mb} MB upload limit.")
        else:
            upload_key = f"{uploaded.name}:{uploaded.size}"
            if st.session_state.get("input_key") != upload_key:
                try:
                    st.session_state.input_frame = read_csv(uploaded)
                    st.session_state.input_source = uploaded.name
                    st.session_state.input_key = upload_key
                    st.session_state.pop("pipeline_result", None)
                except InputValidationError as exc:
                    st.error(str(exc))

    frame = st.session_state.get("input_frame")
    if frame is None:
        st.write("Upload a CSV or load the included 200-row synthetic dataset to begin.")
        return
    st.caption(f"Source: {st.session_state.get('input_source', 'Uploaded CSV')}")
    try:
        records, report = validate_dataframe(frame)
    except InputValidationError as exc:
        st.error(str(exc))
        return
    st.session_state.records = records
    st.session_state.input_metrics = derive_input_metrics(frame)
    metrics = st.columns(4)
    metrics[0].metric("Total rows", report.total_rows)
    metrics[1].metric("Valid rows", report.valid_rows)
    metrics[2].metric("Blank comments", report.blank_comments)
    metrics[3].metric("Duplicate IDs", report.duplicate_ids)
    if "vendor" not in frame.columns:
        st.warning(f"This file has no vendor column. Vendor analysis will use ‘{DEFAULT_VENDOR}’.")
    if report.issues:
        with st.expander("Validation issues", expanded=True):
            st.dataframe([issue.model_dump() for issue in report.issues], width="stretch")

    if not settings.has_credentials:
        st.error("OPENROUTER_API_KEY is not available to the application process. Add it to .env and restart Streamlit.")
    if st.button("Analyse all returns", type="primary", width="stretch", disabled=not records or not settings.has_credentials):
        progress = st.progress(0, text="Starting analysis…")
        pipeline = build_pipeline(settings)
        result = pipeline.run(records, lambda done, total: progress.progress(done / total, text=f"Analysed {done} of {total}"))
        progress.empty()
        st.session_state.pipeline_result = result
        st.success("Analysis complete. Results are shown below and on the Returns Intelligence page.")

    result = st.session_state.get("pipeline_result")
    if result is not None:
        source_metrics = st.session_state.input_metrics
        source_cards = st.columns(2)
        source_cards[0].metric("Overall return rate", format_rate(source_metrics.overall_return_rate), help=source_metrics.return_rate_basis)
        source_cards[1].metric("Reasons captured as ‘Other’", format_rate(source_metrics.other_reason_rate), help="Calculated from non-blank return_reason values in this dataset.")
        counts = {status: sum(item.processing_status.value == status for item in result.results) for status in ["ACCEPTED", "HUMAN_REVIEW", "FAILED"]}
        cols = st.columns(5)
        cols[0].metric("Records", len(result.results))
        cols[1].metric("Accepted", counts["ACCEPTED"])
        cols[2].metric("Human review", counts["HUMAN_REVIEW"])
        cols[3].metric("Failed", counts["FAILED"])
        cols[4].metric("Time", f"{result.processing_seconds:.1f}s")
        if counts["FAILED"]:
            st.warning(
                f"{counts['FAILED']} record(s) could not be classified. Configure the AI service to resolve "
                "‘Other’ comments, then run the analysis again."
            )

        analysed = results_frame(result.results)
        render_reason_distribution(analysed)
        render_sku_vendor_breakdown(analysed)
        render_other_analysis(analysed)

        if st.button("Open full Returns Intelligence dashboard", width="stretch"):
            st.session_state.pending_navigation = "Returns Intelligence"
            st.rerun()
