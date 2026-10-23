"""Initial Streamlit dashboard shell."""

import httpx
import streamlit as st
from app.core.config import get_settings

from frontend.api_client import ApiClient

settings = get_settings()
configured_key = settings.ui_api_key or settings.write_api_key or settings.read_api_key
client = ApiClient(api_key=configured_key.get_secret_value() if configured_key else None)

st.set_page_config(page_title=settings.app_name, page_icon="🛡️", layout="wide")
st.title(settings.app_name)
st.caption(
    "Local development foundation — privacy-aware retrieval and verification "
    "are being added incrementally."
)

overview, documents, assistant, evaluation, training = st.tabs(
    ["Overview", "Documents", "Knowledge assistant", "Evaluation", "Training"]
)
with overview:
    st.metric("Service", "Available")
    st.info(
        "Phase 1 foundation is running. Connect the API with the documented PowerShell command."
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
with evaluation:
    st.subheader("Offline evaluation")
    st.caption("Results are actual deterministic pipeline measurements, not mock scores.")
    if st.button("Run synthetic_v1 evaluation"):
        try:
            run = client.run_evaluation()
            st.success(f"Completed run {run['run_id']}.")
            st.json(run["metrics"])
            failures = [item for item in run["outcomes"] if item["status"] != "completed"]
            if failures:
                st.warning("Some examples failed or were skipped.")
                st.dataframe(failures, use_container_width=True)
        except (httpx.HTTPError, RuntimeError, ValueError) as exc:
            st.error(f"Evaluation failed: {exc}")
    try:
        runs = client.list_evaluations()
        if runs:
            st.dataframe(
                [
                    {
                        "run_id": run["run_id"],
                        "dataset": run["dataset_version"],
                        "failure_rate": run["metrics"].get("failure_rate"),
                        "duration_ms": run["duration_ms"],
                    }
                    for run in runs
                ],
                use_container_width=True,
            )
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        st.info(f"Start the FastAPI service to view evaluation runs: {exc}")
with training:
    st.subheader("Optional LoRA / QLoRA training")
    st.warning(
        "Training downloads a selected model and can consume GPU memory. "
        "No training starts without explicit confirmation."
    )
    try:
        preflight = client.training_preflight()
        st.json(preflight)
        dataset = st.selectbox("Dataset", ["intent_v1"])
        method = st.selectbox("Method", ["lora", "qlora"])
        model = st.text_input("Base model", "sshleifer/tiny-gpt2")
        confirm = st.checkbox("I explicitly confirm model download/training")
        if st.button("Start training"):
            result = client.start_training(dataset, model, method, confirm)
            st.info(f"Training job: {result['job_id']}")
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        st.info(f"Training services unavailable: {exc}")
