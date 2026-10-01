from __future__ import annotations

import logging

import streamlit as st

from src.config import ROOT, Settings
from src.ui import dashboard, methodology, overview, review

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
st.set_page_config(page_title="Dhaga Returns Intelligence", page_icon="🧵", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1400px;}
  [data-testid="stMetric"] {background: #fff7ed; border: 1px solid #fed7aa; padding: 1rem; border-radius: .75rem;}
  h1, h2, h3 {color: #7c2d12;}
</style>
""", unsafe_allow_html=True)

settings = Settings.load()
pending_navigation = st.session_state.pop("pending_navigation", None)
if pending_navigation is not None:
    st.session_state.navigation = pending_navigation
page = st.sidebar.radio("Navigate", ["Analyse Returns", "Returns Intelligence", "Human Review", "Method & Cost"], key="navigation")
st.sidebar.caption("Dhaga & Co.")

if page == "Analyse Returns":
    overview.render(settings, ROOT / "data" / "demo_returns.csv")
elif page == "Returns Intelligence":
    dashboard.render()
elif page == "Human Review":
    review.render()
else:
    methodology.render()
