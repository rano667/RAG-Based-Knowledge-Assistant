"""Small offline benchmark dataset for invoice RAG regression checks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkCase:
    """Expected retrieval and answer signals for one evaluation question."""

    query: str
    expected_keywords: tuple[str, ...]
    expected_chunk_ids: tuple[int, ...]
    ground_truth: str


BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        query="What items are in invoice 0012820?",
        expected_keywords=(
            "Exterior Protection",
            "Temporary Lighting",
            "Theater and Stage Equipment",
        ),
        expected_chunk_ids=(8,),
        ground_truth=(
            "Invoice 0012820 includes Exterior Protection, Temporary Lighting, "
            "and Theater and Stage Equipment."
        ),
    ),
    BenchmarkCase(
        query="What are the products in invoice 1213?",
        expected_keywords=("Glossostigma", "Bayberry", "Waxflower"),
        expected_chunk_ids=(5,),
        ground_truth="Invoice 1213 includes Glossostigma, Bayberry, and Waxflower.",
    ),
)

# Kept as a readable alias for existing notebooks or scripts.
eval_data = BENCHMARK_CASES
