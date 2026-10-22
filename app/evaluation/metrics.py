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
