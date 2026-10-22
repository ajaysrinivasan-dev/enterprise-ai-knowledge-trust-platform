"""Versioned API routes for the platform foundation."""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.ingestion.parsers import parse_document
from app.ingestion.validation import validate_upload

router = APIRouter(prefix="/api/v1")


class EvidenceInput(BaseModel):
    """Evidence supplied to deterministic verification."""

    chunk_id: str
    document_id: str
    source_name: str
    text: str
    score: float = 1.0


class VerificationRequest(BaseModel):
    """Request for claim-level answer verification."""

    answer: str
    evidence: list[EvidenceInput] = Field(default_factory=list)


class ClaimResponse(BaseModel):
    """Public claim verification result."""

    claim: str
    status: str
    evidence_ids: list[str]
    confidence: float
    explanation: str


class VerificationResponse(BaseModel):
    """Public verification response."""

    status: str
    method: str
    claims: list[ClaimResponse]


class DocumentResponse(BaseModel):
    """Safe indexed-document metadata."""

    document_id: str
    source_name: str
    pii_counts: dict[str, int]
    chunk_count: int


class RetrievalResponse(BaseModel):
    """Retrieved evidence with actual chunk identifiers."""

    chunk_id: str
    document_id: str
    source_name: str
    text: str
    score: float


class QuestionRequest(BaseModel):
    """Question and retrieval controls."""

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.25, ge=0.0, le=1.0)


class QuestionResponse(BaseModel):
    """Grounded answer response."""

    answer: str
    abstained: bool
    citations: list[dict[str, str]]
    evidence: list[RetrievalResponse]
    reason: str | None = None


class AgentRequest(BaseModel):
    """Validated workflow request."""

    intent: str
    question: str | None = None
    answer: str | None = None
    recipient: str | None = None
    subject: str | None = None
    body: str | None = None
    report: str | None = None
    evidence: list[EvidenceInput] = Field(default_factory=list)
    approved: bool = False


class AgentResponse(BaseModel):
    """Safe workflow output with execution stages."""

    intent: str | None
    result: object | None
    approval_required: bool
    trace: list[str]
    error: str | None


class EvaluationRequest(BaseModel):
    """Validated offline evaluation request."""

    dataset_id: str = Field(pattern=r"^[a-z0-9_]+$")
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.25, ge=0.0, le=1.0)


class EvaluationRunResponse(BaseModel):
    """Public evaluation result including per-example outcomes."""

    run_id: str
    dataset_version: str
    created_at: str
    metrics: dict[str, float | None]
    errors: list[str]
    config: dict[str, object]
    outcomes: list[dict[str, object]]
    duration_ms: float


class EvaluationCompareResponse(BaseModel):
    """Metric deltas for compatible runs."""

    baseline_run_id: str
    candidate_run_id: str
    dataset_version: str
    deltas: dict[str, float | None]


class HealthResponse(BaseModel):
    """Public health response without internal paths or secrets."""

    status: str
    service: str
    environment: str
    timestamp: datetime


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report that the application process is serving requests."""

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.environment,
        timestamp=datetime.now(UTC),
    )


@router.get("/ready", response_model=HealthResponse, tags=["system"])
def readiness(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    """Report foundation readiness; dependency checks will be added by later phases."""

    return HealthResponse(
        status="ready",
        service=settings.app_name,
        environment=settings.environment,
        timestamp=datetime.now(UTC),
    )


@router.post("/documents", response_model=DocumentResponse, tags=["documents"])
async def upload_document(
    file: Annotated[UploadFile, File(...)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DocumentResponse:
    """Validate, parse, redact, and index a document without exposing its contents."""

    filename = file.filename or ""
    content = await file.read()
    try:
        extension = validate_upload(filename, content, settings.max_upload_bytes)
        parsed = parse_document(filename, content, extension)
        from app.api.rag_runtime import index_document

        metadata = index_document(parsed)
    except (ValueError, RuntimeError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DocumentResponse.model_validate(metadata)


@router.get("/documents", response_model=list[DocumentResponse], tags=["documents"])
def documents() -> list[DocumentResponse]:
    """List indexed document metadata."""

    from app.api.rag_runtime import list_documents

    return [DocumentResponse.model_validate(item) for item in list_documents()]


@router.post("/retrieve", response_model=list[RetrievalResponse], tags=["retrieval"])
def retrieve(request: QuestionRequest) -> list[RetrievalResponse]:
    """Inspect retrieved evidence without generating an answer."""

    from app.api.rag_runtime import retrieve as retrieve_evidence

    return [
        RetrievalResponse(
            chunk_id=result.chunk.chunk_id,
            document_id=result.chunk.document_id,
            source_name=result.chunk.source_name,
            text=result.chunk.text,
            score=result.score,
        )
        for result in retrieve_evidence(request.question, request.top_k, request.min_score)
    ]


@router.post("/questions", response_model=QuestionResponse, tags=["generation"])
def question(request: QuestionRequest) -> QuestionResponse:
    """Return a grounded retrieval-only answer with validated source citations."""

    from app.api.rag_runtime import answer
    from app.api.rag_runtime import retrieve as retrieve_evidence

    result = answer(request.question, request.top_k, request.min_score)
    evidence = retrieve_evidence(request.question, request.top_k, request.min_score)
    return QuestionResponse(
        answer=result.answer,
        abstained=result.abstained,
        citations=[
            {
                "chunk_id": citation.chunk_id,
                "source_name": citation.source_name,
                "excerpt": citation.excerpt,
            }
            for citation in result.citations
        ],
        evidence=[
            RetrievalResponse(
                chunk_id=item.chunk.chunk_id,
                document_id=item.chunk.document_id,
                source_name=item.chunk.source_name,
                text=item.chunk.text,
                score=item.score,
            )
            for item in evidence
        ],
        reason=result.reason,
    )


@router.post("/agent", response_model=AgentResponse, tags=["agents"])
def agent(request: AgentRequest) -> AgentResponse:
    """Execute a bounded workflow route using the shared retrieval runtime."""

    from app.agents.workflow import AgentWorkflow
    from app.api.rag_runtime import retrieve as retrieve_evidence

    workflow = AgentWorkflow(retrieve_evidence)
    result = workflow.run(request.model_dump(exclude_none=True), approved=request.approved)
    return AgentResponse(
        intent=result.intent.value if result.intent else None,
        result=result.result,
        approval_required=result.approval_required,
        trace=list(result.trace),
        error=result.error,
    )


def _evaluation_response(run) -> EvaluationRunResponse:
    return EvaluationRunResponse(
        run_id=run.run_id,
        dataset_version=run.dataset_version,
        created_at=run.created_at.isoformat(),
        metrics=run.metrics,
        errors=list(run.errors),
        config=run.config.__dict__,
        outcomes=[outcome.__dict__ for outcome in run.outcomes],
        duration_ms=run.duration_ms,
    )


@router.post("/evaluations", response_model=EvaluationRunResponse, tags=["evaluation"])
def run_evaluation(
    request: EvaluationRequest,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Run an approved offline evaluation dataset."""

    from app.api.evaluation_runtime import run_evaluation as execute
    from app.evaluation.models import EvaluationConfig

    try:
        run = execute(
            request.dataset_id,
            settings.evaluation_database,
            config=EvaluationConfig(top_k=request.top_k, min_score=request.min_score),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _evaluation_response(run)


@router.get("/evaluations", response_model=list[EvaluationRunResponse], tags=["evaluation"])
def list_evaluations(settings: Annotated[Settings, Depends(get_settings)]):
    """List stored evaluation runs."""

    from app.api.evaluation_runtime import evaluation_store

    store = evaluation_store(settings.evaluation_database)
    return [_evaluation_response(run) for run in store.list_runs()]


@router.get("/evaluations/{run_id}", response_model=EvaluationRunResponse, tags=["evaluation"])
def get_evaluation(run_id: str, settings: Annotated[Settings, Depends(get_settings)]):
    """Retrieve one stored evaluation run."""

    from app.api.evaluation_runtime import get_run

    try:
        return _evaluation_response(get_run(run_id, settings.evaluation_database))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/evaluations/{baseline_id}/compare/{candidate_id}",
    response_model=EvaluationCompareResponse,
)
def compare_evaluations(
    baseline_id: str,
    candidate_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Compare metrics only when dataset and pipeline configuration match."""

    from app.api.evaluation_runtime import get_run

    try:
        baseline = get_run(baseline_id, settings.evaluation_database)
        candidate = get_run(candidate_id, settings.evaluation_database)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if baseline.compatible_key != candidate.compatible_key:
        raise HTTPException(status_code=409, detail="evaluation runs are not compatible")
    keys = set(baseline.metrics) | set(candidate.metrics)
    deltas = {
        key: (
            candidate.metrics.get(key) - baseline.metrics.get(key)
            if candidate.metrics.get(key) is not None and baseline.metrics.get(key) is not None
            else None
        )
        for key in keys
    }
    return EvaluationCompareResponse(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        dataset_version=baseline.dataset_version,
        deltas=deltas,
    )


@router.post("/verify", response_model=VerificationResponse, tags=["verification"])
def verify(request: VerificationRequest) -> VerificationResponse:
    """Verify answer claims against caller-provided retrieved evidence."""

    from app.retrieval.models import DocumentChunk, RetrievalResult
    from app.verification.service import verify_answer

    evidence = [
        RetrievalResult(
            chunk=DocumentChunk(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                source_name=item.source_name,
                text=item.text,
            ),
            score=item.score,
        )
        for item in request.evidence
    ]
    result = verify_answer(request.answer, evidence)
    return VerificationResponse(
        status=result.status.value,
        method=result.method,
        claims=[
            ClaimResponse(
                claim=claim.claim,
                status=claim.status.value,
                evidence_ids=list(claim.evidence_ids),
                confidence=claim.confidence,
                explanation=claim.explanation,
            )
            for claim in result.claims
        ],
    )
