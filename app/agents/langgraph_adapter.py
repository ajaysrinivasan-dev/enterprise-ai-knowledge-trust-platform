"""Optional LangGraph adapter for the explicit EAKTP workflow."""

from typing import TypedDict

from app.agents.workflow import AgentWorkflow


class GraphState(TypedDict, total=False):
    """Serializable state exchanged by optional LangGraph nodes."""

    request: dict
    approved: bool
    result: object
    error: str | None
    trace: list[str]


def build_langgraph(workflow: AgentWorkflow):
    """Build a bounded graph when the optional LangGraph package is installed."""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise RuntimeError("LangGraph support requires the optional 'agents' dependency") from exc

    graph = StateGraph(GraphState)

    def execute(state: GraphState) -> GraphState:
        result = workflow.run(state["request"], approved=state.get("approved", False))
        return {
            "result": result.result,
            "error": result.error,
            "trace": list(result.trace),
        }

    graph.add_node("execute", execute)
    graph.add_edge(START, "execute")
    graph.add_edge("execute", END)
    return graph.compile()
