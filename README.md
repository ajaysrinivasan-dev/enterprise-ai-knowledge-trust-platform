# Enterprise AI Knowledge & Trust Platform

EAKTP is a local-first, privacy-aware NLP platform for document-grounded answers,
claim verification, controlled tools, evaluation, and optional model fine-tuning.

## Current status

The current implementation includes modular configuration and logging, FastAPI
health/readiness routes, privacy-aware document upload and indexing, deterministic
offline embeddings/retrieval, optional Sentence Transformers and ChromaDB adapters,
grounded retrieval-only answers, optional Ollama generation, citation validation, and a
Streamlit API client. Claim verification, bounded workflow routing, allowlisted SQL
reports, approval-gated email drafts, and deterministic evaluation metrics are also
implemented.
See [PROJECT_STATUS.md](./PROJECT_STATUS.md) for verified work and the next phase.

## Quick start (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,documents,privacy]"
Copy-Item .env.example .env
python -m pytest
python -m ruff check .
python -m uvicorn app.main:app --reload
```

In another PowerShell window:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run frontend\streamlit_app.py
```

For persistent vector storage and local embedding models, install the optional RAG
group:

```powershell
python -m pip install -e ".[rag]"
```

The runtime honors `EAKTP_EMBEDDING_BACKEND` (`hash` by default, or
`sentence-transformers`) and `EAKTP_VECTOR_BACKEND` (`sqlite` by default, `memory`, or
`chroma`). SQLite stores sanitized chunks, embeddings, and safe document metadata in
`EAKTP_SQLITE_VECTOR_PATH` (default `data/runtime/eaktp.sqlite3`), so retrieval and the
document list survive API restarts without an external service. The memory backend is
intentionally ephemeral. ChromaDB stores vectors/chunks persistently, but the current
runtime's restart-safe document catalog is implemented for SQLite; with Chroma selected,
document metadata remains process-local. Sentence Transformers may download its configured
model if it is not cached. Use a separate vector database/path when changing embedding
models because vector dimensions and spaces may differ. Runtime SQLite files are local data
and should remain excluded from version control.

For the optional LangGraph adapter:

```powershell
python -m pip install -e ".[agents]"
```

For optional evaluation and training workflows:

```powershell
python -m pip install -e ".[evaluation,training]"
```

The deterministic evaluation fixture is `data\evaluation\synthetic_v1.jsonl`, and the
intent-routing benchmark fixture is `data\training\intent_v1.jsonl`. Evaluation runs
record a stable dataset fingerprint and synthetic-data status; the current test suite
does not claim RAGAS, DeepEval, LoRA, or QLoRA production results.

## Optional training workflow

The training subsystem validates `data\training\intent_v1.jsonl`, creates seeded
partitions, performs runtime hardware checks, and exposes explicit background training
and adapter benchmark endpoints. It does not download models or start training
automatically. Install the optional dependencies only when you intend to run training:

```powershell
python -m pip install -e ".[training]"
python -c "from app.training.preflight import collect_preflight; import json; print(json.dumps(collect_preflight().as_dict(), indent=2))"
```

Start training only after confirming the preflight and dataset are suitable:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/training/runs `
  -ContentType "application/json" `
  -Body '{"dataset_id":"intent_v1","base_model":"sshleifer/tiny-gpt2","method":"lora","confirm":true}'
```

QLoRA requires runtime CUDA, compatible bitsandbytes, and sufficient VRAM. Use the
isolated GPU environment only for an explicitly approved smoke test; the five-example
synthetic dataset is only a smoke-test fixture and cannot support production
generalization claims. Check the
selected Hugging Face model card and dataset terms before downloading or training; the
default `sshleifer/tiny-gpt2` configuration is intended for a local smoke test only.

Before a confirmed training request is submitted, the API requires non-empty train,
validation, and test partitions. The bundled five-example fixture is therefore rejected
for real training because its validation split is empty; it remains available for
deterministic smoke tests. Failed jobs are not registered as adapters, and incomplete
run directories are cleaned up. Benchmark comparisons also require matching dataset
fingerprints for the held-out test split.

Run the approved offline evaluation through the API after starting FastAPI:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/api/v1/evaluations `
  -ContentType "application/json" `
  -Body '{"dataset_id":"synthetic_v1","top_k":5,"min_score":0.25}'
```

Evaluation runs persist under `data/runtime/evaluation.db` by default. The API accepts
dataset identifiers, not arbitrary filesystem paths. Results distinguish completed and
failed examples, record actual errors, latency, dataset fingerprints, and dataset
status, and expose only deterministic metrics unless an external evaluator is separately
integrated and executed.

Set `EAKTP_OLLAMA_MODEL` in `.env` only after confirming the model exists locally with
`ollama list`. Without it, the API returns explicitly labeled retrieval-only evidence
and never fabricates an LLM answer.

Open the API documentation at `http://127.0.0.1:8000/docs` and the dashboard at
`http://localhost:8501`. Python 3.11–3.14 is supported by the project metadata; use
the interpreter available on the machine, subject to dependency compatibility.

## Security and privacy

The API supports Bearer-key access control. Configure these variables in your ignored
`.env` file; never commit real secrets. Use separate, randomly generated secrets of at
least 32 characters for each enabled role:

- `EAKTP_API_KEY`: optional legacy administrator key. Production-like environments
  (anything other than `development`, `dev`, or `test`) require either this key or an
  administrator entry in `EAKTP_API_PRINCIPALS_JSON`. Administrators can use all API routes.
- `EAKTP_READ_API_KEY`: optional reader key. Permits read-only HTTP methods (`GET`, `HEAD`,
  `OPTIONS`) on non-admin-only routes.
- `EAKTP_WRITE_API_KEY`: optional writer key. Permits normal read/write API routes, but
  not administrator-only training routes, the `/api/v1/agent` route, or any evaluation
  route. Evaluation runs include per-example outcomes and are administrator-only.
- `EAKTP_UI_API_KEY`: tenant-scoped credential the Streamlit frontend uses for normal document
  and question operations. It is required when `EAKTP_API_PRINCIPALS_JSON` is configured and must
  match a reader or writer principal. A reader cannot upload. Without the registry, a blank UI
  key falls back to the legacy writer, then reader key; it never implicitly uses the admin key.
- `EAKTP_ADMIN_UI_API_KEY`: optional separate credential used only for administrator-only
  evaluation, training, and agent operations. With a principal registry, it must match a registry
  entry whose role is `admin`; without the registry, it must exactly match `EAKTP_API_KEY`. The UI
  does not fall back to a writer credential, and the backend continues to enforce administrator
  permissions. If the registry has no admin principal, add one under authorized administration
  before enabling these UI operations; do not promote a writer key.

When at least one key is configured, requests to `/api/v1/*` must include
`Authorization: Bearer <key>`, except `/api/v1/health` and `/api/v1/ready`, which remain
public. A blank role-specific key disables that key; do not reuse the same secret across
roles. These legacy role keys map reader/writer traffic to the shared `default` tenant;
they do not separate individual users. For tenant isolation, configure
`EAKTP_API_PRINCIPALS_JSON` with unique credentials mapped to a principal, tenant, and role.
The principal registry is static configuration, not an identity provider; store real keys
only in `.env` or a secret manager. When using this registry, set `EAKTP_UI_API_KEY` to the
intended tenant-scoped frontend credential; a shared Streamlit process otherwise acts as its
configured service principal, not as each browser user. Tenant IDs come from this server-side
mapping, never from request bodies or headers. Security logs emit structured events for
authentication failures, authorization denials, and authenticated API request outcomes;
they exclude bearer keys and request bodies, but are not a tamper-proof or complete audit trail.
The API also applies a configurable in-process per-client-IP limit (default 120 requests per
60 seconds) using `EAKTP_API_RATE_LIMIT_REQUESTS` and
`EAKTP_API_RATE_LIMIT_WINDOW_SECONDS`. This is a single-process fallback, not a distributed
limit; a production ingress or API gateway must enforce its own limits.

Example JSON value for `EAKTP_API_PRINCIPALS_JSON` (replace every placeholder with a
unique random secret of at least 32 characters):

```json
[
  {"key":"<unique-secret-for-alice-32-chars-min>","principal_id":"alice","tenant_id":"tenant-a","role":"writer"},
  {"key":"<unique-secret-for-bob-32-chars-min>","principal_id":"bob","tenant_id":"tenant-b","role":"writer"},
  {"key":"<unique-admin-secret-32-chars-min>","principal_id":"platform-admin","tenant_id":"admin-home","role":"admin"}
]
```

The SQLite, in-memory, and Chroma retrieval paths filter document listing and retrieval by
tenant. Existing SQLite rows are migrated with a NULL tenant and remain invisible to
non-admin tenants; administrators can inspect these legacy rows. Admin uploads are placed
in the configured admin home tenant (or `default` for the legacy admin key). Chroma's
existing metadata catalog remains process-local. This is a single-process static-principal
mechanism, not full SSO, account lifecycle, fine-grained document ACLs, or an audit system.
Document deletion is tenant-scoped and returns the same not-found response for a document
outside the caller's scope. Upload bodies are read only up to one byte beyond the configured
maximum before size validation. Principal JSON is validated during settings/application
startup; malformed, duplicate, short, or ambiguously assigned principals fail without
including credential values in the error.

Example PowerShell request using the administrator key:

```powershell
$headers = @{ Authorization = "Bearer $env:EAKTP_API_KEY" }
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/documents -Headers $headers
```

Keep the service bound to localhost unless you have also configured TLS, network controls,
key rotation, rate limiting, and appropriate operational monitoring. This application does not
terminate TLS or implement a distributed rate limiter: for a network deployment, terminate TLS
at a trusted reverse proxy or ingress, enforce request/body/time limits (including chunked
uploads without `Content-Length`) and per-client or per-credential rate limits there, and
restrict direct access to the application port. Do not
trust forwarded client-IP headers unless the proxy is explicitly configured as trusted. Rotate
keys by generating a replacement secret, updating the server-side secret store, restarting or
rolling the service, validating the replacement, then revoking the old key; avoid logging or
committing either key. Store production secrets in a secret manager rather than source control
or container images. Uploads are redacted before indexing, but pattern-based PII redaction is
not guaranteed to catch every sensitive value.

## Current limitations

The default runtime remains local and process-scoped. ChromaDB, Sentence Transformers,
Ollama, LangGraph, RAGAS/DeepEval, and training libraries are optional and are not
required for the deterministic test suite. The deterministic verifier is a conservative
fallback, not proof of factual correctness. Email drafts never send automatically. Role-key authentication is available as described above; when no keys are configured in local
development, API routes remain open for convenient testing. The Streamlit process uses one
configured service credential for all browser sessions; it does not provide per-browser-user
authentication or isolation.

See [docs/architecture.md](./docs/architecture.md) and the attached implementation
requirements for the phased RAG, verification, agent, evaluation, and training work.
