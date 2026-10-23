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

- Authentication and full multi-user persistence are not implemented.
- Readiness currently checks only process-level readiness; dependency checks will be
  added when those dependencies exist.

## Integration and release-readiness audit

**Status: verified offline; production integrations remain opt-in**

Fixed and verified:

- Question responses use one retrieval result set for generated citations and displayed
  evidence.
- Evaluation datasets reject empty or duplicate case IDs and expose stable fingerprints.
- Persisted evaluation runs record dataset fingerprint and synthetic/approved status;
  comparisons include these compatibility fields.
- Training benchmarks require a non-empty held-out test split, a fingerprint, and an
  explicit dataset status.
- Ingestion redacts PII before chunking and indexing; workflow SQL remains read-only and
  allowlisted; email remains draft-only.

Verification:

- `py -m pytest`: 55 passed.
- `py -m ruff check .`: passed.
- `py -m compileall -q app tests`: passed.

Remaining blockers:

- No approved production intent or evaluation dataset is present.
- Optional Sentence Transformers, ChromaDB, Ollama, LangGraph, external evaluators, and
  real production-scale training remain unverified.

## Phase 4 — Claim verification and agent workflows

**Status: core complete**

Implemented files:

- `app/verification/models.py`
- `app/verification/service.py`
- `app/agents/models.py`
- `app/agents/workflow.py`
- `app/agents/langgraph_adapter.py`
- `app/tools/sql.py`
- `app/api/routes.py`
- `tests/test_verification_agents.py`

Implemented behavior:

- Atomic claim extraction and structured claim results.
- Supported, unsupported, contradicted, and uncertain classifications.
- Conservative contradiction detection for explicit negation and conflicting numeric facts.
- Replaceable judge interface with validation of status, confidence, and evidence IDs.
- Invalid judge responses and provider failures become uncertain, never supported.
- Typed bounded workflow routes for question answering, document search, SQL reports,
  email drafts, and verification.
- Explicit validation, routing, retrieval/generation/verification stages, timeout checks,
  bounded retry configuration, safe traces, and approval-gated email drafts.
- Optional LangGraph adapter under the `agents` dependency group.
- SQL reports are allowlisted, read-only, parameterized where inputs exist, row-bounded,
  and timeout-bounded.
- `POST /api/v1/agent` exposes safe traces and approval status.

Verification:

- `py -m pytest`: 24 passed.
- `py -m ruff check .`: passed.
- LangGraph and real model-backed judge integrations were not executed because the
  optional package/service is not installed or configured.

## Phase 5 — Deterministic evaluation foundation

**Status: complete for offline evaluation; optional judge adapters unexecuted**

Implemented files:

- `app/evaluation/models.py`
- `app/evaluation/metrics.py`
- `app/evaluation/dataset.py`
- `app/evaluation/storage.py`
- `data/evaluation/synthetic_v1.jsonl`
- `tests/test_evaluation.py`

Implemented behavior:

- Versioned synthetic JSONL evaluation fixture.
- Deterministic Recall@k, Precision@k, binary F1, and route accuracy metrics.
- Deterministic answer reference exact-match metric explicitly labeled as lexical, plus
  citation validity and citation relevance metrics.
- SQLite evaluation-run persistence with timestamps, dataset versions, metrics, and errors.
- Empty/invalid dataset validation.
- Reproducible `EvaluationRunner` using the existing retrieval, generation, and
  verification services.
- Per-example completed/failed outcomes, error summaries, latency, configuration,
  completion/failure rates, and aggregate metrics.
- Approved dataset API endpoints:
  - `POST /api/v1/evaluations`
  - `GET /api/v1/evaluations`
  - `GET /api/v1/evaluations/{run_id}`
  - `GET /api/v1/evaluations/{baseline_id}/compare/{candidate_id}`
- Streamlit evaluation section showing actual run metrics, failures, and run history.
- Optional RAGAS and DeepEval availability checks with explicit missing-dependency errors.

Verification:

- `py -m pytest`: 26 passed.
- `py -m pytest`: 32 passed after runner/API integration and metric expansion.
- `py -m ruff check .`: passed.

Limitations:

- RAGAS/DeepEval execution adapters are not yet implemented; only availability checks
  exist, and no external evaluator ran.
- No evaluation score is claimed for production or real company data.

## Next phase

The evaluation milestone is complete for the deterministic offline path. The next
milestone should implement the optional LoRA/QLoRA training workflow, including safe
train/validation/test splitting, hardware checks, adapter persistence, and measured
base-vs-fine-tuned comparison. Training foundation schemas already exist but no training
job has been run.

## Phase 6 — Training and benchmark foundation

**Status: core foundation complete**

Implemented files:

- `app/training/dataset.py`
- `app/training/benchmark.py`
- `data/training/intent_v1.jsonl`
- `tests/test_training.py`
- `pyproject.toml` optional `training` dependency group

Implemented behavior:

- Versioned intent-routing JSONL fixture covering all supported workflow intents.
- Duplicate and invalid-label detection to reduce train/test leakage risk.
- Provider-neutral benchmark result schema with accuracy, sample count, latency, and
  explicit `trained` status.
- CPU-only benchmark smoke path that does not download models or train at runtime.
- Optional Transformers, PEFT, TRL, and datasets dependencies are declared but not
  installed or executed in this environment.

Verification:

- `py -m pytest`: 28 passed.
- `py -m ruff check .`: passed.

Remaining:

- Training/validation/test splitting, LoRA/QLoRA training script, hardware checks,
  adapter persistence, and a real base-vs-fine-tuned experiment.
- Evaluation runner/API/dashboard integration and optional RAGAS/DeepEval adapters.

## Phase 7 — LoRA/QLoRA workflow

**Status: implemented and tested; no real training run executed**

Implemented files:

- `app/training/dataset.py`
- `app/training/config.py`
- `app/training/preflight.py`
- `app/training/artifacts.py`
- `app/training/lora.py`
- `app/training/benchmark.py`
- `app/api/training_runtime.py`
- `app/api/routes.py`
- `frontend/api_client.py`
- `frontend/streamlit_app.py`
- `tests/test_training.py`
- `tests/test_training_artifacts.py`
- `tests/test_training_api.py`
- `tests/test_training_benchmark.py`
- `pyproject.toml`

Implemented behavior:

- Schema, label, empty-input, duplicate, near-duplicate, and class-distribution checks.
- Reproducible seeded train/validation/test splitting without duplicating examples.
- Dataset fingerprints and warnings for insufficient class sizes or empty validation data.
- Runtime Python, PyTorch, CUDA, GPU, VRAM, disk, and optional-package preflight.
- Explicit LoRA and QLoRA configurations with validation.
- Lazy Transformers/PEFT training path with bounded `max_steps`, conservative defaults,
  explicit confirmation, prompt-token loss masking, separate adapter output, tokenizer,
  metadata, dataset fingerprint, and split seed.
- QLoRA is rejected unless CUDA, sufficient detected VRAM, and bitsandbytes are present.
- Safe adapter metadata loading, base-model/method validation, artifact existence checks,
  and approved-root listing.
- Baseline benchmark with accuracy, per-class precision/recall/F1, macro-F1,
  weighted-F1, confusion matrix, latency, failure rate, and sample count.
- Same-held-out-set baseline-versus-adapter benchmark endpoint.
- Single-process bounded background training job with duplicate-run prevention.
- API endpoints:
  - `GET /api/v1/training/preflight`
  - `GET /api/v1/training/datasets`
  - `POST /api/v1/training/runs`
  - `GET /api/v1/training/runs/{job_id}`
  - `GET /api/v1/training/adapters`
  - `POST /api/v1/training/benchmarks`
- Streamlit training section with preflight display and explicit confirmation.

Actual preflight executed:

- Platform: Windows 11
- Python: 3.14.7
- PyTorch: 2.14.1+cpu
- CUDA: unavailable
- GPU: unavailable at runtime
- Free disk: approximately 129.0 GB
- Transformers: available
- PEFT: unavailable
- bitsandbytes: unavailable
- datasets: unavailable

Consequences:

- No LoRA or QLoRA training run completed.
- No adapter was saved or reloaded.
- No baseline-versus-fine-tuned metrics were measured.
- The tiny synthetic dataset contains 5 examples, produces train=4, validation=0,
  test=1, and is only a smoke-test fixture—not evidence of generalization.

Opt-in run requirements:

```powershell
python -m pip install -e ".[training]"
python -c "from app.training.preflight import collect_preflight; print(collect_preflight().as_dict())"
```

For QLoRA, use a verified CUDA-enabled environment with compatible bitsandbytes and
enough VRAM; the current CPU-only environment cannot run it. Training must be started
explicitly through the API/dashboard and is not automatic.

## Phase 8 — Pre-training readiness audit

**Status: audited; no training run, model download, package installation, or method
change performed**

Verified and corrected:

- The five-example `intent_v1` fixture remains available for deterministic tests and is
  explicitly treated as a smoke-test dataset.
- Real training now rejects empty validation/test partitions and insufficient training
  data before a background job is submitted. It does not fabricate or duplicate rows.
- The API returns an actionable 422 response for the smoke fixture because validation is
  empty.
- Training now checks tokenizer padding, rejects tokenizers without pad/EOS tokens,
  selects supported attention target modules from the loaded architecture, and fails
  clearly for unsupported causal-language-model architectures.
- Adapter loading validates local tokenizer files and the PEFT-recorded base model
  against adapter metadata.
- Failed training jobs clean up their incomplete run directory; incomplete runs are not
  listed as usable adapters because only completed metadata is accepted.
- Benchmark comparisons now require matching held-out dataset fingerprints in addition
  to matching sample counts. The API stamps both baseline and adapter results with the
  exact test-split fingerprint.
- Runtime and API output directories are aligned with `EAKTP_TRAINING_OUTPUT_DIR`.
- No arbitrary pickle, joblib, dill, cloudpickle, or `torch.load` deserialization was
  found in application or training tests. Adapter metadata is JSON-only and model
  loading uses Transformers/PEFT local adapter APIs.
- `.gitignore` excludes `artifacts/`, `models/`, `data/runtime/`, SQLite databases,
  caches, and environment secrets.

Checks executed:

- `py -m pytest`: **42 passed**, with two existing dependency deprecation warnings.
- `py -m ruff check .`: **passed**.
- Static inspection of dataset splitting, model/tokenizer preparation, loss masking,
  label normalization, adapter lifecycle, benchmark formulas, background jobs, API
  routes, Streamlit controls, and repository ignore rules.
- Static search for unsafe serialized-object loading.

Only mocked or inspected:

- Target-module selection, adapter reload, tokenizer behavior, model generation,
  interrupted-job cleanup, and incompatible-architecture handling were covered by
  deterministic guards and inspection, not a real model execution.
- Baseline/fine-tuned benchmark arithmetic was tested with deterministic objects, but no
  real fine-tuned model result exists.
- Streamlit was inspected; no browser session was launched.

Still unverified:

- A genuine LoRA training run and adapter reload.
- A genuine QLoRA run on CUDA with bitsandbytes.
- Real base-model inference and measured baseline-versus-adapter performance.
- Production-scale dataset quality or generalization.

## Phase 9 — Configurable retrieval provider wiring

**Status: implemented and offline-tested; optional providers not executed**

Implemented:

- Runtime retrieval now honors `EAKTP_EMBEDDING_BACKEND` (`hash` or
  `sentence-transformers`) and `EAKTP_VECTOR_BACKEND` (`memory` or `chroma`).
- Hash embeddings and in-memory storage remain the no-download defaults.
- Explicit optional-provider configuration fails clearly if its dependency is missing;
  there is no silent fallback to a different embedding/vector backend.
- The vector store is initialized before a Sentence Transformers model, so missing
  ChromaDB fails before a model could be initialized or downloaded.
- `.env.example` and README document backend settings, model/cache implications, and the
  need for separate Chroma paths when changing embedding models.
- Added tests for default backend selection and configured optional-provider wiring using
  fakes; no optional service or model was downloaded or started.

Verification after this milestone:

- `py -m pytest`: 58 passed, with two existing dependency deprecation warnings.
- `py -m ruff check .`: passed after fixing import ordering in the new tests.
- `py -m compileall -q app tests`: passed.

Remaining limitations:

- Persistent Chroma vectors/chunks do not yet make the document-list metadata persistent;
  `_documents` is still process-local.
- ChromaDB and Sentence Transformers were not installed/loaded or exercised against a real
  model/service in this milestone.
- An approved production intent dataset is still required; no synthetic records were
  promoted or invented as production data.

## Phase 10 — Persistent document catalog and SQLite retrieval

**Status: implemented and verified**

Verification: `py -m pytest` — 59 passed (two existing dependency deprecation warnings); `py -m ruff check .` — passed; `py -m compileall -q app tests` — passed.

Implemented:

- Added a built-in SQLite vector store that persists sanitized chunks and their embeddings,
  allowing retrieval to survive API process restarts without an external service.
- Stores safe document metadata (document ID, source name, PII detection counts, chunk count)
  alongside vectors in the same SQLite database. Document listings filter out records whose
  chunk counts do not match persisted chunks.
- SQLite is now the default vector backend; the explicit in-memory backend remains available
  for tests/ephemeral development and Chroma remains optional.
- Added document metadata validation against stored chunk counts and coordinated deletion of
  chunks plus catalog metadata.
- Added a restart-style test that creates a document with an email address, reconstructs the
  store from the same database, confirms retrieval/listing survive, checks raw email is absent,
  and verifies deletion removes both chunks and metadata.
- Updated `.env.example` and README to document backend selection and the remaining Chroma
  catalog limitation.

Important limitations:

- Restart-safe catalog persistence is currently supported by SQLite. If Chroma is selected,
  its vectors persist but this runtime's document metadata catalog remains process-local.
- The database stores redacted chunk text and embeddings; it does not preserve original
  uploaded files or raw unredacted text.
- This is a local single-node persistence layer, not a multi-user or distributed database.


## Phase 11 — Shared API-key authentication

**Status: implemented; offline verification in progress**

Implemented:

- Added optional `EAKTP_API_KEY` as a secret setting. When configured, API requests to `/api/v1/*` require an exact `Authorization: Bearer ...` credential, compared using constant-time comparison.
- `/api/v1/health` and `/api/v1/ready` remain public for monitoring.
- Non-development environments refuse to initialize without a non-empty API key of at least 32 characters.
- Blank API keys remain disabled in development so copying `.env.example` does not unexpectedly break local requests.
- Updated `.env.example` and README with key setup, PowerShell usage, and explicit security limitations.
- Added tests for missing/wrong/correct keys, public health endpoints, production configuration requirements, and blank development keys.

Security boundary:

- This is shared-key authentication only; it does not provide individual identities, roles, per-document access control, rate limiting, key rotation, or TLS. Keep the server bound to localhost until deployment controls are configured.


## Phase 12 — Tenant-aware document isolation

**Status: implemented and offline-tested; static principal configuration only**

Implemented:

- Added optional `EAKTP_API_PRINCIPALS_JSON` server-side mapping from unique bearer keys to `principal_id`, `tenant_id`, and role. Principal keys must be at least 32 characters; duplicate keys and non-admin `*` tenant scopes are rejected. Request-supplied tenant IDs are not used for authorization.
- Preserved the existing role-key variables for compatibility. Legacy reader/writer keys share the `default` tenant; the legacy administrator key has all-tenant read access and writes to `default`.
- Added tenant ownership metadata to chunks and indexed-document records, and tenant-scoped listing/search for SQLite, memory, and Chroma retrieval backends.
- Added an SQLite schema migration that leaves existing records with NULL ownership. Those records are hidden from tenant-scoped reads and remain accessible only through the administrator all-tenant scope.
- Scoped document upload, listing, retrieval inspection, question evidence, and citations using the authenticated request context.
- Updated the Streamlit API client to attach configured bearer credentials; it uses `EAKTP_UI_API_KEY` when supplied, otherwise the writer then reader key, and does not implicitly elevate to the administrator key. Documented principal-registry setup and limitations.
- Added cross-tenant API tests, an SQLite legacy-schema migration test, and a SQLite tenant-filtering test.

Verification:

- `py -m pytest`: 70 passed, with two existing Starlette/AnyIO deprecation warnings.
- `py -m ruff check .`: passed.
- `py -m compileall -q app tests frontend`: passed.

Limitations and deployment notes:

- The principal registry is static configuration, not SSO or a full identity provider. It does not provide account lifecycle, fine-grained per-document ACLs, key rotation, rate limiting, or a complete audit trail.
- Legacy shared reader/writer keys intentionally share one `default` tenant and therefore do not isolate users from one another; use distinct entries in `EAKTP_API_PRINCIPALS_JSON` for tenant isolation.
- Admin uploads use the configured admin principal's home tenant, or `default` for the legacy admin key; admin reads can inspect all tenants and legacy unowned rows.
- Chroma's document metadata catalog remains process-local even though tenant metadata is attached to stored chunks. SQLite remains the recommended default for restart-safe catalog persistence.
- Optional Chroma behavior was implemented but not exercised against a live ChromaDB installation in this verification.

## Phase 13 — Security hardening and enterprise-readiness audit

**Status: implemented and offline-verified**

Fixed and verified:

- Evaluation listing, retrieval, comparison, and execution are administrator-only because
  persisted runs include per-example outcomes and are not tenant-owned.
- Principal registry JSON is validated while constructing settings, including valid JSON,
  required fields, minimum key length, unique credentials and principal IDs, valid roles,
  and safe tenant scope rules. Error messages do not include credential values.
- Document deletion is exposed through a tenant-scoped API route. Cross-tenant or unknown
  document IDs return the same `404` response, while administrators retain explicit
  all-tenant deletion access.
- Upload reads are bounded to the configured maximum plus one byte before validation.
  Failed metadata persistence removes already-indexed chunks to avoid partial documents.
- Regression coverage includes evaluation authorization, startup configuration validation,
  cross-tenant deletion, oversized uploads, tenant retrieval/listing, and SQLite legacy
  ownership migration.

Verification:

- Focused security, tenant, lifecycle, and evaluation tests: 23 passed.
- Full-suite, Ruff, and compile results are recorded after the final validation run below.

Known limitations:

- ChromaDB was not installed or exercised live; its tenant filtering remains covered by
  adapter-level code and existing tests only where dependencies are available.
- Streamlit uses its configured service credential for every browser session and does not
  implement per-browser identity or isolation.
- Static bearer credentials still have no rotation, rate limiting, SSO, or complete audit
  trail.

## Phase 14 — Follow-up security audit and integration hardening

**Status: local hardening implemented; live ChromaDB verification blocked by missing build tooling**

Follow-up findings fixed:

- Production settings now accept a validated administrator principal from
  EAKTP_API_PRINCIPALS_JSON without requiring a duplicate legacy EAKTP_API_KEY.
  Effective principal-ID collisions with implicit legacy identities are rejected.
- Fixed the in-memory retrieval backend's administrator wildcard handling. An admin
  principal with tenant_id="*" can now retrieve across tenants, matching SQLite and
  Chroma adapter behavior. Added a regression assertion.
- Added structured security events for failed authentication, denied authorization,
  rate-limit events, oversized declared uploads, and authenticated request outcomes.
  Logs do not include bearer credentials or request bodies.
- Added a configurable in-process per-client-IP API rate limiter
  (EAKTP_API_RATE_LIMIT_REQUESTS, default 120;
  EAKTP_API_RATE_LIMIT_WINDOW_SECONDS, default 60) with HTTP 429 and Retry-After.
  This is a fallback only; distributed enforcement belongs at the ingress/API gateway.
- Added an early Content-Length guard for clearly oversized multipart upload requests
  before Starlette's multipart parser runs. The route still reads at most the configured
  file limit plus one byte. Chunked requests without Content-Length must be capped at
  the reverse proxy/ingress.
- Broadened indexing cleanup so indexed chunks are removed if metadata persistence raises
  any ordinary exception, and added a regression test.
- Documented key-rotation steps, secret-manager use, TLS termination, ingress body/time
  limits, distributed rate limiting, structured-log limitations, and Streamlit's shared
  service identity boundary.

Verification:

- py -m pytest -q: **78 passed**, two existing Starlette/AnyIO deprecation warnings.
- py -m ruff check .: passed.
- py -m compileall -q app tests frontend: passed.
- git diff --check: passed (only Git LF/CRLF working-copy warnings).
- Added adapter-level Chroma tenant-filter regression coverage using a fake collection.
- Attempted isolated ChromaDB installation under Python 3.14 and 3.12. Both attempts stopped
  because chroma-hnswlib requires Microsoft Visual C++ 14.0 or newer to build on this
  machine. No project dependency files were changed by those attempts, and no live Chroma
  database test was possible.

Remaining risks:

- Live Chroma persistence, filtering, and deletion are still not verified against the actual
  ChromaDB runtime. Install Microsoft C++ Build Tools or use an environment with a compatible
  prebuilt chroma-hnswlib wheel, then rerun isolated integration tests.
- The in-process limiter is per-process and keyed by the ASGI client address; it is not a
  substitute for a distributed ingress/API-gateway limiter, especially behind a proxy.
- TLS termination, production secret-manager integration, alerting, durable/tamper-resistant
  audit storage, SSO, and per-browser Streamlit identity require deployment/infrastructure
  work. Keep the API bound to localhost until those controls are configured.
- Pattern-based PII detection remains best-effort.

No commits, pushes, deployments, production data changes, or external service calls were made.
Existing worktree changes were preserved.
