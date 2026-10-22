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

For the optional LangGraph adapter:

```powershell
python -m pip install -e ".[agents]"
```

For optional evaluation and training workflows:

```powershell
python -m pip install -e ".[evaluation,training]"
```

The deterministic evaluation fixture is `data\evaluation\synthetic_v1.jsonl`, and the
intent-routing benchmark fixture is `data\training\intent_v1.jsonl`. The current test
suite measures deterministic behavior only; it does not claim RAGAS, DeepEval, LoRA, or
QLoRA results.

Set `EAKTP_OLLAMA_MODEL` in `.env` only after confirming the model exists locally with
`ollama list`. Without it, the API returns explicitly labeled retrieval-only evidence
and never fabricates an LLM answer.

Open the API documentation at `http://127.0.0.1:8000/docs` and the dashboard at
`http://localhost:8501`. Python 3.11–3.14 is supported by the project metadata; use
the interpreter available on the machine, subject to dependency compatibility.

## Security and privacy assumptions

The initial app is for local development and has no user authentication. Do not expose
it to a company network. Secrets belong in `.env`, which is ignored by Git. Later
phases must validate uploads, redact PII before indexing, restrict tools, and avoid
logging raw prompts or documents.

## Current limitations

The default runtime remains local and process-scoped. ChromaDB, Sentence Transformers,
Ollama, LangGraph, RAGAS/DeepEval, and training libraries are optional and are not
required for the deterministic test suite. The deterministic verifier is a conservative
fallback, not proof of factual correctness. Email drafts never send automatically, and
the local app has no authentication.

See [docs/architecture.md](./docs/architecture.md) and the attached implementation
requirements for the phased RAG, verification, agent, evaluation, and training work.
