# Enterprise AI Knowledge & Trust Platform

EAKTP is a local-first, privacy-aware NLP platform for document-grounded answers,
claim verification, controlled tools, evaluation, and optional model fine-tuning.

## Current status

Phase 1 and the deterministic core of Phase 2 are implemented: modular project
structure, validated environment settings, structured logging, FastAPI health/readiness
endpoints, a Streamlit shell, safe upload validation, text parsing, and PII redaction.
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

Open the API documentation at `http://127.0.0.1:8000/docs` and the dashboard at
`http://localhost:8501`. Python 3.11–3.14 is supported by the project metadata; use
the interpreter available on the machine, subject to dependency compatibility.

## Security and privacy assumptions

The initial app is for local development and has no user authentication. Do not expose
it to a company network. Secrets belong in `.env`, which is ignored by Git. Later
phases must validate uploads, redact PII before indexing, restrict tools, and avoid
logging raw prompts or documents.

## Planned components

See [docs/architecture.md](./docs/architecture.md) and the attached implementation
requirements for the phased RAG, verification, agent, evaluation, and training work.
