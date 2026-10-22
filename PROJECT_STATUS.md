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
