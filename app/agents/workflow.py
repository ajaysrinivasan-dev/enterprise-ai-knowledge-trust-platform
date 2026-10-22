"""Explicit bounded workflow orchestration.

The graph is deliberately implemented without requiring LangGraph at runtime so the
normal test suite remains deterministic. The nodes and conditional transitions map
directly to a LangGraph StateGraph when the optional dependency is installed.
"""

from collections.abc import Callable
from time import monotonic

from app.agents.models import AgentResult, AgentState, Intent
from app.generation.service import generate_grounded_answer
from app.retrieval.models import DocumentChunk, RetrievalResult
from app.tools.email import create_email_draft
from app.tools.sql import ReadOnlySqlTool
from app.verification.service import verify_answer


class AgentWorkflow:
    """Run approved routes with bounded retries and no side-effecting actions."""

    def __init__(
        self,
        retrieve: Callable[[str, int, float], list],
        *,
        generate: Callable[[str, list[str]], str] | None = None,
        sql_tool: ReadOnlySqlTool | None = None,
        max_retries: int = 2,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.retrieve = retrieve
        self.generate = generate
        self.sql_tool = sql_tool
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

    def run(self, request: dict, *, approved: bool = False) -> AgentResult:
        """Execute a request through explicit validation, routing, and finalization."""

        started = monotonic()
        state = AgentState(request=request, approved=approved)
        try:
            self._validate(state)
            self._route(state)
            while state.trace[-1] != "finalize":
                if monotonic() - started > self.timeout_seconds:
                    raise TimeoutError("workflow timed out")
                self._dispatch(state)
        except (RuntimeError, TimeoutError, ValueError) as exc:
            state.error = str(exc)
            state.trace.append("error")
        return AgentResult(
            intent=state.intent,
            result=state.result,
            approval_required=state.approval_required,
            trace=tuple(state.trace),
            error=state.error,
        )

    def _validate(self, state: AgentState) -> None:
        state.trace.append("validate_input")
        if not isinstance(state.request.get("intent"), str):
            raise ValueError("intent is required")
        try:
            state.intent = Intent(state.request["intent"])
        except ValueError as exc:
            raise ValueError("unsupported workflow intent") from exc

    def _route(self, state: AgentState) -> None:
        state.trace.append("route")
        state.approval_required = state.intent is Intent.EMAIL_DRAFT

    def _dispatch(self, state: AgentState) -> None:
        if state.intent in {Intent.QUESTION, Intent.SEARCH}:
            self._retrieve(state)
        elif state.intent is Intent.VERIFY:
            self._verify(state)
        elif state.intent is Intent.SQL_REPORT:
            self._sql(state)
        elif state.intent is Intent.EMAIL_DRAFT:
            self._email(state)
        state.trace.append("finalize")

    def _retrieve(self, state: AgentState) -> None:
        state.trace.append("retrieve")
        question = str(state.request.get("question", "")).strip()
        if not question:
            raise ValueError("question is required")
        state.evidence = self.retrieve(
            question,
            int(state.request.get("top_k", 5)),
            float(state.request.get("min_score", 0.25)),
        )
        if state.intent is Intent.SEARCH:
            state.result = state.evidence
            return
        state.trace.append("generate")
        if self.generate is None:
            def generator(_question: str, evidence: list[str]) -> str:
                return "Relevant evidence:\n" + "\n".join(f"- {item}" for item in evidence)
        else:
            generator = self.generate
        state.answer = generate_grounded_answer(
            question, state.evidence, generator, minimum_score=float(
                state.request.get("min_score", 0.25)
            )
        )
        state.result = state.answer

    def _verify(self, state: AgentState) -> None:
        state.trace.append("verify")
        answer = str(state.request.get("answer", ""))
        raw_evidence = state.request.get("evidence", [])
        evidence = [
            item
            if isinstance(item, RetrievalResult)
            else RetrievalResult(
                DocumentChunk(
                    chunk_id=str(item["chunk_id"]),
                    document_id=str(item["document_id"]),
                    source_name=str(item["source_name"]),
                    text=str(item["text"]),
                ),
                float(item.get("score", 1.0)),
            )
            for item in raw_evidence
        ]
        state.verification = verify_answer(answer, evidence)
        state.result = state.verification

    def _sql(self, state: AgentState) -> None:
        state.trace.append("sql")
        if self.sql_tool is None:
            raise RuntimeError("SQL tool is not configured")
        state.result = self.sql_tool.run(str(state.request.get("report", "")))

    def _email(self, state: AgentState) -> None:
        state.trace.append("approval")
        if not state.approved:
            state.result = create_email_draft(
                str(state.request.get("recipient", "")),
                str(state.request.get("subject", "")),
                str(state.request.get("body", "")),
            )
            return
        raise RuntimeError("email sending is not supported; approval cannot send email")
