"""Lightweight custom metrics for offline RAG evaluation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

WORD_RE = re.compile(r"\b[\w'-]+\b")


def keyword_recall(answer: str, expected_keywords: Iterable[str]) -> float:
    """Return the fraction of expected keywords present in an answer."""

    expected = tuple(expected_keywords)
    if not expected:
        return 0.0

    normalized_answer = answer.casefold()
    matches = sum(keyword.casefold() in normalized_answer for keyword in expected)
    return matches / len(expected)


def keyword_match_count(answer: str, expected_keywords: Iterable[str]) -> int:
    """Count expected keyword phrases found in an answer."""

    normalized_answer = answer.casefold()
    return sum(keyword.casefold() in normalized_answer for keyword in expected_keywords)


def word_overlap_grounding_score(answer: str, context: str) -> float:
    """Estimate grounding by answer-token overlap with retrieved context."""

    answer_tokens = [token.casefold() for token in WORD_RE.findall(answer)]
    if not answer_tokens:
        return 0.0

    context_tokens = {token.casefold() for token in WORD_RE.findall(context)}
    grounded_tokens = sum(token in context_tokens for token in answer_tokens)
    return grounded_tokens / len(answer_tokens)


def precision_at_k(
    retrieved_ids: Sequence[int | None],
    expected_ids: Iterable[int],
    k: int,
) -> float:
    """Compute retrieval precision for the first `k` returned chunk IDs."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    expected = set(expected_ids)
    retrieved_top_k = retrieved_ids[:k]
    relevant = sum(chunk_id in expected for chunk_id in retrieved_top_k)
    return relevant / k


def has_possible_hallucination(grounding_score: float, threshold: float = 0.45) -> bool:
    """Flag low-overlap answers for manual review, not as a final truth label."""

    return grounding_score < threshold


# Compatibility aliases for earlier experiments.
keyword_match_score = keyword_match_count
grounding_score = word_overlap_grounding_score
