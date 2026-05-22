"""Offline benchmark runner.

Run with:
    python -m app.evaluation.run_eval
    python -m app.evaluation.run_eval --ragas
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from app.config import Settings, get_settings
from app.evaluation.benchmark import BENCHMARK_CASES, BenchmarkCase
from app.evaluation.metrics import (
    has_possible_hallucination,
    keyword_match_count,
    keyword_recall,
    precision_at_k,
    word_overlap_grounding_score,
)
from app.evaluation.ragas_eval import RagasSample, run_ragas_batch
from app.evaluation.report import (
    CaseResult,
    EvaluationSummary,
    build_summary,
    print_custom_report,
)
from app.logger import configure_logging, log_event
from app.rag import RagPipeline, build_rag_pipeline

logger = logging.getLogger(__name__)


def evaluate_custom_metrics(
    pipeline: RagPipeline,
    cases: Sequence[BenchmarkCase] = BENCHMARK_CASES,
) -> tuple[EvaluationSummary, list[RagasSample]]:
    """Run offline benchmark cases through the RAG pipeline."""

    custom_results: list[CaseResult] = []
    ragas_samples: list[RagasSample] = []

    for case in cases:
        response = pipeline.ask(case.query)
        grounding = word_overlap_grounding_score(response.answer, response.context)
        matches = keyword_match_count(response.answer, case.expected_keywords)

        custom_results.append(
            CaseResult(
                query=case.query,
                answer=response.answer,
                retrieved_chunk_ids=response.retrieved_chunk_ids,
                retrieved_sources=response.retrieved_sources,
                precision_at_1=precision_at_k(
                    response.retrieved_chunk_ids,
                    case.expected_chunk_ids,
                    k=1,
                ),
                keyword_recall=keyword_recall(response.answer, case.expected_keywords),
                keyword_matches=matches,
                expected_keyword_count=len(case.expected_keywords),
                grounding_score=grounding,
                possible_hallucination=has_possible_hallucination(grounding),
                latency_seconds=response.latency_seconds,
            )
        )
        ragas_samples.append(
            RagasSample(
                question=case.query,
                answer=response.answer,
                contexts=[response.context],
                ground_truth=case.ground_truth,
            )
        )

    return build_summary(custom_results), ragas_samples


def run_offline_evaluation(
    settings: Settings | None = None,
    include_ragas: bool = False,
) -> EvaluationSummary:
    """Build an offline pipeline and print selected evaluation reports."""

    active_settings = settings or get_settings()
    configure_logging(active_settings.log_level, active_settings.json_logs)
    log_event(logger, "offline_evaluation_started", include_ragas=include_ragas)

    summary, ragas_samples = evaluate_custom_metrics(build_rag_pipeline(active_settings))
    print_custom_report(summary)

    if include_ragas:
        print("\nRAGAS EVALUATION")
        print("=" * 48)
        print(run_ragas_batch(ragas_samples, active_settings))

    log_event(logger, "offline_evaluation_completed", cases=len(summary.cases))
    return summary


def parse_args() -> argparse.Namespace:
    """Parse evaluation CLI flags."""

    parser = argparse.ArgumentParser(description="Run the offline RAG benchmark.")
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Also run LLM-based RAGAS metrics after custom metrics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_offline_evaluation(include_ragas=parse_args().ragas)
