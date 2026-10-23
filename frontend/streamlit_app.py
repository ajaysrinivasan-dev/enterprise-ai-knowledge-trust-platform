"""Initial Streamlit dashboard shell."""

import streamlit as st
from app.core.config import get_settings

from frontend.api_client import (
    ApiAuthenticationError,
    ApiAuthorizationError,
    ApiClient,
    ApiClientError,
    ApiConnectionError,
)
from frontend.auth import (
    FrontendCredentialError,
    resolve_frontend_admin_api_key,
    resolve_frontend_api_key,
)

settings = get_settings()
try:
    client = ApiClient(
        api_key=resolve_frontend_api_key(settings),
        admin_api_key=resolve_frontend_admin_api_key(settings),
    )
    credential_error = ""
except FrontendCredentialError as exc:
    client = None
    credential_error = str(exc)

st.set_page_config(page_title=settings.app_name, page_icon="🛡️", layout="wide")
st.title(settings.app_name)
st.caption(
    "Local development foundation — privacy-aware retrieval and verification "
    "are being added incrementally."
)
if client is None:
    st.error(credential_error)
    st.stop()

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
    if client is None:
        st.error(credential_error)
    else:
        upload = st.file_uploader("Upload a TXT, Markdown, PDF, or DOCX document")
        if upload is not None and st.button("Index document"):
            try:
                result = client.upload_document(upload.name, upload.getvalue())
                st.success(f"Indexed {result['source_name']} ({result['chunk_count']} chunks).")
                st.json(result["pii_counts"])
            except ApiAuthenticationError as exc:
                st.error(str(exc))
            except ApiAuthorizationError as exc:
                st.error(str(exc))
            except ApiConnectionError as exc:
                st.error(str(exc))
            except ApiClientError as exc:
                st.error(str(exc))
        try:
            st.dataframe(client.list_documents(), use_container_width=True)
        except ApiAuthenticationError as exc:
            st.error(str(exc))
        except ApiAuthorizationError as exc:
            st.error(str(exc))
        except ApiConnectionError as exc:
            st.info(str(exc))
        except ApiClientError as exc:
            st.error(str(exc))
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
        except (ApiClientError, ValueError) as exc:
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
        except (ApiClientError, ValueError) as exc:
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
    except (ApiClientError, ValueError) as exc:
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
    except (ApiClientError, ValueError) as exc:
        st.info(f"Training services unavailable: {exc}")
