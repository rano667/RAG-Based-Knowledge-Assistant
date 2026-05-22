"""Optional RAGAS evaluation integration for offline runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from app.config import Settings, get_settings


@dataclass(frozen=True)
class RagasSample:
    """RAGAS-ready evaluation sample."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


def run_ragas_batch(
    samples: Sequence[RagasSample],
    settings: Settings | None = None,
) -> Any:
    """Evaluate benchmark samples with RAGAS using Groq and HF embeddings."""

    if not samples:
        raise ValueError("RAGAS evaluation requires at least one sample.")

    # Lazy imports keep the API startup path independent of evaluation packages.
    from datasets import Dataset
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    active_settings = settings or get_settings()
    dataset = Dataset.from_dict(
        {
            "question": [sample.question for sample in samples],
            "answer": [sample.answer for sample in samples],
            "contexts": [sample.contexts for sample in samples],
            "ground_truth": [sample.ground_truth for sample in samples],
        }
    )

    return evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ChatGroq(model=active_settings.evaluator_model, temperature=0),
        embeddings=HuggingFaceEmbeddings(model_name=active_settings.embedding_model),
    )


def run_ragas_evaluation(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> Any:
    """Compatibility wrapper for single-sample RAGAS experiments."""

    return run_ragas_batch(
        [
            RagasSample(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth,
            )
        ]
    )
