"""Optional RAGAS and DeepEval adapters."""


def check_ragas() -> str:
    """Report whether RAGAS is importable without claiming an execution."""

    try:
        import ragas  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("RAGAS is not installed; install the evaluation extra") from exc
    return "available-but-not-executed"


def check_deepeval() -> str:
    """Report whether DeepEval is importable without invoking a judge."""

    try:
        import deepeval  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("DeepEval is not installed; install the evaluation extra") from exc
    return "available-but-not-executed"
