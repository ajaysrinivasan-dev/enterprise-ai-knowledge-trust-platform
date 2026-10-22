# Project status

## Phase 1 — Project foundation

**Status: complete**

Implemented:

- Modular Python package and frontend/data/docs/scripts directories.
- Pydantic settings with `.env` support and upload/model configuration placeholders.
- Structured logging configuration that does not log document contents.
- FastAPI application factory with restricted development CORS.
- Versioned `/api/v1/health` and `/api/v1/ready` endpoints.
- Streamlit dashboard shell with explicit not-yet-implemented feature messaging.
- `pytest`, Ruff, packaging metadata, `.gitignore`, README, and architecture notes.
- API tests covering safe health/readiness responses.

Verification:

- `py -m pytest`: 6 passed.
- `py -m ruff check .`: passed.

## Phase 2 — Document ingestion and privacy

**Status: core complete**

Implemented:

- Safe basename and path-traversal rejection, extension allowlist, size limits, and
  empty-content checks.
- UTF-8 text, Markdown, PDF, and DOCX parser adapters. PDF/DOCX adapters fail
  explicitly with an actionable dependency error when optional packages are absent.
- Typed parsed-document and PII result models.
- Deterministic email, phone, SSN, labeled-name, and labeled-address detection with
  offsets and confidence values.
- Stable document-scoped redaction placeholders and aggregate counts that exclude raw
  values from metadata.
- Tests for traversal, size limits, empty documents, text normalization, detection, and
  redaction.

Verification:

- `py -m pytest`: 6 passed.
- `py -m ruff check .`: passed.

Limitations:

- The HTTP upload endpoint, persistent document metadata, Presidio adapter, and indexing
  integration are deferred to later phases.
- Pattern detection is probabilistic and does not guarantee that all PII is found.

## Phase 3 — Retrieval and grounded generation

**Status: core complete; optional provider verification pending**

Implemented:

- Metadata-preserving word-window chunking with validated overlap and stable chunk IDs.
- Dependency-free in-memory lexical retriever for local fallback and deterministic tests.
- Retrieval result objects that always retain the originating document and source name.
- Deterministic hash embedding provider for offline tests.
- Optional Sentence Transformer embedding provider.
- Persistent ChromaDB vector-store adapter with safe upsert and document deletion.
- In-memory cosine vector store for offline API operation.
- Optional cross-encoder reranker.
- Ollama provider with configurable URL/model/timeout and explicit failure handling.
- Privacy-aware indexing that redacts PII before embedding.
- Grounded answer orchestration with real chunk citations and threshold-based abstention.
- API document upload/listing, retrieval inspection, and question endpoints.
- Streamlit remains a shell; API-backed upload and question controls are next.

Verification:

- `py -m pytest`: 9 passed.
- `py -m ruff check .`: passed after formatting the new test imports.

Limitations:

- Optional Sentence Transformer, ChromaDB, cross-encoder, and Ollama integrations were
  implemented but not verified here because model downloads/services are unavailable.

## Next phase

The next implementation slice will add Streamlit API integration, document deletion,
provider configuration/readiness reporting, and then continue with LangGraph workflows.

## Current additional checkpoint

Implemented:

- Grounded retrieval-only answer objects with explicit abstention.
- Citation construction and validation against actual retrieval results.
- Deterministic claim extraction and supported/unsupported classification.
- Allowlisted read-only SQLite reporting with bounded rows.
- Email draft creation that always requires approval and never sends.
- `POST /api/v1/verify` with structured claim-level responses.

Verification:

- `py -m pytest`: 15 passed.
- `py -m ruff check .`: passed.

Limitations:

- Verification is a deterministic lexical fallback, not an NLI model or LLM judge.
- The current API runtime uses process-local metadata and an in-memory vector store by
  default; ChromaDB persistence is available through the optional RAG dependency group.
- The current API answer path is retrieval-only unless an Ollama-backed generator is
  explicitly wired; no model-generated answer is claimed as verified.
- LangGraph, evaluation, training, and the complete Streamlit workflow remain outstanding.

Additional RAG integration:

- `app/core/config.py` now exposes Ollama, embedding, vector-backend, and Chroma path
  settings.
- `app/api/rag_runtime.py` uses Ollama when `EAKTP_OLLAMA_MODEL` is configured and
  otherwise uses an explicitly labeled retrieval-only generator.
- `frontend/api_client.py` and `frontend/streamlit_app.py` provide API-backed upload,
  document listing, and question/citation display.
- `.env.example` and `README.md` document the optional RAG configuration.

Latest verification:

- `py -m pytest`: 19 passed.
- `py -m ruff check .`: passed.
- Real Sentence Transformer, ChromaDB, cross-encoder, and Ollama calls were not run;
  they require optional packages, model downloads, and/or a running local service.

## Known limitations

- Authentication, persistence, document ingestion, retrieval, generation, verification,
  agent tools, evaluation, and fine-tuning are not implemented yet.
- Readiness currently checks only process-level readiness; dependency checks will be
  added when those dependencies exist.
