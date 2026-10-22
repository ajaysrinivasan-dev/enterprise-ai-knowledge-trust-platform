"""Initial Streamlit dashboard shell."""

import streamlit as st
from app.core.config import get_settings

settings = get_settings()

st.set_page_config(page_title=settings.app_name, page_icon="🛡️", layout="wide")
st.title(settings.app_name)
st.caption(
    "Local development foundation — privacy-aware retrieval and verification "
    "are being added incrementally."
)

overview, documents, assistant = st.tabs(["Overview", "Documents", "Knowledge assistant"])
with overview:
    st.metric("Service", "Available")
    st.info(
        "Phase 1 foundation is running. Connect the API with the documented "
        "PowerShell command."
    )
with documents:
    st.subheader("Document management")
    st.warning("Document upload and PII processing arrive in Phase 2.")
with assistant:
    st.subheader("AI knowledge assistant")
    st.warning("Retrieval and grounded answers arrive in Phase 3.")
