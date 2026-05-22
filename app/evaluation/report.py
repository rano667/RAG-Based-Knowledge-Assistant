"""Evaluation result structures and console reporting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseResult:
    """Custom metric output for one benchmark case."""

    query: str
    answer: str
    retrieved_chunk_ids: list[int | None]
    retrieved_sources: list[str | None]
    precision_at_1: float
    keyword_recall: float
    keyword_matches: int
    expected_keyword_count: int
    grounding_score: float
    possible_hallucination: bool
    latency_seconds: float


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregated custom evaluation metrics."""

    cases: list[CaseResult]
    average_precision_at_1: float
    average_keyword_recall: float
    average_grounding_score: float
    possible_hallucinations: int
    average_latency_seconds: float


def build_summary(cases: list[CaseResult]) -> EvaluationSummary:
    """Aggregate case-level metrics into a report."""

    if not cases:
        raise ValueError("Cannot summarize an empty evaluation run.")

    total = len(cases)
    return EvaluationSummary(
        cases=cases,
        average_precision_at_1=sum(case.precision_at_1 for case in cases) / total,
        average_keyword_recall=sum(case.keyword_recall for case in cases) / total,
        average_grounding_score=sum(case.grounding_score for case in cases) / total,
        possible_hallucinations=sum(case.possible_hallucination for case in cases),
        average_latency_seconds=sum(case.latency_seconds for case in cases) / total,
    )


def print_custom_report(summary: EvaluationSummary) -> None:
    """Print a compact offline evaluation report."""

    print("\nCUSTOM RAG EVALUATION")
    print("=" * 48)
    for index, case in enumerate(summary.cases, start=1):
        print(f"\nCase {index}: {case.query}")
        print(f"Retrieved chunks: {case.retrieved_chunk_ids}")
        print(f"Sources: {case.retrieved_sources}")
        print(f"Precision@1: {case.precision_at_1:.2f}")
        print(
            "Keyword recall: "
            f"{case.keyword_recall:.2f} "
            f"({case.keyword_matches}/{case.expected_keyword_count})"
        )
        print(f"Word-overlap grounding: {case.grounding_score:.2f}")
        print(f"Possible hallucination: {case.possible_hallucination}")
        print(f"Latency: {case.latency_seconds:.2f} sec")
        print("Answer:")
        print(case.answer)

    print("\nSUMMARY")
    print("-" * 48)
    print(f"Average Precision@1: {summary.average_precision_at_1:.2f}")
    print(f"Average keyword recall: {summary.average_keyword_recall:.2f}")
    print(f"Average word-overlap grounding: {summary.average_grounding_score:.2f}")
    print(f"Possible hallucinations: {summary.possible_hallucinations}")
    print(f"Average latency: {summary.average_latency_seconds:.2f} sec")


# Compatibility wrapper for the first evaluation split.
def print_final_report(results: EvaluationSummary) -> None:
    """Print an evaluation summary."""

    print_custom_report(results)
