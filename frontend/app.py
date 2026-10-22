"""APAC-Face | Facial Landmark Intelligence - Streamlit dashboard.

    streamlit run frontend/app.py

Inference goes through the FastAPI backend (APAC_API_URL, default http://127.0.0.1:8000);
if the backend is not running the models are loaded in-process instead.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="APAC-Face | Facial Landmark Intelligence", page_icon=":material/face:",
                   layout="wide", initial_sidebar_state="expanded")

import common  # noqa: E402  (must follow set_page_config)

common.apply_style()

pages = [
    st.Page("views/home.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/detect.py", title="Landmark Detection", icon=":material/center_focus_strong:"),
    st.Page("views/live.py", title="Live Camera", icon=":material/videocam:"),
    st.Page("views/comparison.py", title="Model Comparison", icon=":material/leaderboard:"),
    st.Page("views/robustness.py", title="Robustness Analysis", icon=":material/insights:"),
    st.Page("views/about.py", title="About", icon=":material/info:"),
]

with st.sidebar:
    st.markdown("### APAC-Face")
    st.caption("Facial Landmark Intelligence")
    health = common.api_health()
    if health:
        st.success(f"API online - {len(health.get('models_loaded', []))} model(s) on {health.get('device')}",
                   icon=":material/cloud_done:")
    else:
        st.info("API offline - models run in-process", icon=":material/memory:")
    st.caption(f"Backend: `{common.API_URL}`")

st.navigation(pages).run()
