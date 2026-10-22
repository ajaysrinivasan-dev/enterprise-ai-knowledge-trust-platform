"""Deterministic evaluation metrics independent of model providers."""

from collections.abc import Iterable


def recall_at_k(retrieved: Iterable[str], relevant: Iterable[str], k: int) -> float:
    """Measure the fraction of relevant IDs found in the first k results."""

    retrieved_list = list(retrieved)
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    return len(set(retrieved_list[:k]) & relevant_set) / len(relevant_set)


def precision_at_k(retrieved: Iterable[str], relevant: Iterable[str], k: int) -> float:
    """Measure relevant IDs among the first k retrieved results."""

    selected = list(retrieved)[:k]
    if not selected:
        return 0.0
    return len(set(selected) & set(relevant)) / len(selected)


def binary_f1(predicted: Iterable[bool], expected: Iterable[bool]) -> float:
    """Compute binary F1 with a zero-safe denominator."""

    pairs = list(zip(predicted, expected, strict=True))
    true_positive = sum(pred and actual for pred, actual in pairs)
    false_positive = sum(pred and not actual for pred, actual in pairs)
    false_negative = sum(not pred and actual for pred, actual in pairs)
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 1.0


def routing_accuracy(predicted: Iterable[str], expected: Iterable[str]) -> float:
    """Compute exact-match route accuracy."""

    pairs = list(zip(predicted, expected, strict=True))
    return sum(actual == guess for guess, actual in pairs) / len(pairs) if pairs else 0.0


def answer_correctness(answer: str | None, reference: str | None) -> float | None:
    """Return exact normalized match; None means no reference, not a failed score."""

    if answer is None or reference is None:
        return None
    return float(" ".join(answer.casefold().split()) == " ".join(reference.casefold().split()))


def mean_or_none(values: Iterable[float | None]) -> float | None:
    """Average measured values while excluding unavailable metrics."""

    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def citation_validity(citation_ids: Iterable[str], retrieved_ids: Iterable[str]) -> float | None:
    """Measure citations that reference actual retrieved chunks."""

    citations = list(citation_ids)
    if not citations:
        return None
    retrieved = set(retrieved_ids)
    return sum(citation in retrieved for citation in citations) / len(citations)


def citation_relevance(
    citation_ids: Iterable[str], relevant_ids: Iterable[str]
) -> float | None:
    """Measure cited chunks that are labeled relevant for the case."""

    citations = list(citation_ids)
    if not citations:
        return None
    relevant = set(relevant_ids)
    return sum(citation in relevant for citation in citations) / len(citations)
