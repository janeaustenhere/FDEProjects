from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Method & Cost")
    st.markdown("""
### Cost-aware workflow

**Bulk model → Router/Evaluator → Strong model when needed → Evaluator → Accept or Human Review**

- Existing structured reasons such as Size Issue, Defective Pieces, Not Delivered, and Late Delivery bypass the model.
- Only records marked Other are sent through AI interpretation.
- The bulk model handles ordinary language classification in bounded multi-record requests, with a configurable number of batches running concurrently.
- The router escalates low-confidence, uncertain, contradictory, or multi-issue comments.
- A high-confidence bulk result must pass evaluator verification before it can be accepted.
- If the evaluator rejects a bulk result, the strong model reconsiders it and the new result is evaluated again.
- Final evaluator candidates are processed in configurable batches of 50, with bounded batch parallelism.
- The strong model independently reconsiders difficult comments.
- The evaluator protects aggregate business insight from unsupported classifications.
- Python—not an LLM—calculates counts, percentages, ranking, filters, and cost.
- The final distribution merges structured reasons with AI-resolved Other comments without double counting.

The MVP identifies return drivers. It does not claim to reduce Dhaga’s return rate.
""")
    pipeline = st.session_state.get("pipeline_result")
    if pipeline is None:
        st.info("Run an analysis to see usage, latency, and cost.")
        return
    usage = [item.model_dump() for item in pipeline.usage]
    if usage:
        st.dataframe(usage, width="stretch", hide_index=True)
    else:
        st.caption("No model calls were recorded.")
    total_cost = sum(item.estimated_cost for item in pipeline.usage)
    total_calls = sum(item.calls for item in pipeline.usage)
    cols = st.columns(4)
    cols[0].metric("Model calls", total_calls)
    cols[1].metric("Processing time", f"{pipeline.processing_seconds:.2f}s")
    cols[2].metric("Total run cost", f"${total_cost:.6f}")
    cols[3].metric("Average / return", f"${total_cost / len(pipeline.results):.6f}" if pipeline.results else "$0")
    assumed = st.number_input("Assumed weekly returns", min_value=0, value=0, help="A user-provided scenario—not a verified Dhaga volume.")
    average = total_cost / len(pipeline.results) if pipeline.results else 0
    st.metric("Estimated production cost", f"${average * assumed:,.2f} / week", help="Based on your volume assumption and this run's average model cost.")
