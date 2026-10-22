"""Initial Streamlit dashboard shell."""

import httpx
import streamlit as st
from app.core.config import get_settings

from frontend.api_client import ApiClient

settings = get_settings()
client = ApiClient()

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
    upload = st.file_uploader("Upload a TXT, Markdown, PDF, or DOCX document")
    if upload is not None and st.button("Index document"):
        try:
            result = client.upload_document(upload.name, upload.getvalue())
            st.success(f"Indexed {result['source_name']} ({result['chunk_count']} chunks).")
            st.json(result["pii_counts"])
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            st.error(f"Document indexing failed: {exc}")
    try:
        st.dataframe(client.list_documents(), use_container_width=True)
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        st.info(f"Start the FastAPI service to view indexed documents: {exc}")
with assistant:
    st.subheader("AI knowledge assistant")
    question = st.text_input("Ask about indexed documents")
    if st.button("Ask") and question.strip():
        try:
            result = client.ask(question)
            st.write(result["answer"])
            if result["abstained"]:
                st.warning(result["reason"] or "Insufficient evidence.")
            for citation in result["citations"]:
                with st.expander(f"{citation['source_name']} — {citation['chunk_id']}"):
                    st.write(citation["excerpt"])
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            st.error(f"Question failed: {exc}")
