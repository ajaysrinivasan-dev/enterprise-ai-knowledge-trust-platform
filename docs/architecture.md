# Architecture

EAKTP is a modular monolith. FastAPI exposes versioned HTTP routes, the Streamlit
frontend uses those routes, and domain packages isolate ingestion, privacy, retrieval,
generation, verification, agents, evaluation, training, observability, and storage.

The initial milestone only contains the process boundary, configuration, health routes,
and UI shell. Later phases must preserve this separation and keep external model
providers optional.

