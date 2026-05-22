"""API and pipeline data structures."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Question accepted by the FastAPI endpoint."""

    query: str = Field(min_length=1, description="Question to answer from loaded PDFs.")


class QueryResponse(BaseModel):
    """Public API response for a RAG answer."""

    answer: str
    retrieved_chunk_ids: list[int | None]
    retrieved_sources: list[str | None]
    latency_seconds: float


@dataclass(frozen=True)
class RagResponse:
    """Internal response with evaluation context kept off the public API."""

    answer: str
    retrieved_chunk_ids: list[int | None]
    retrieved_sources: list[str | None]
    context: str
    latency_seconds: float

    def to_api_response(self) -> QueryResponse:
        """Expose user-facing fields without returning full retrieved context."""

        return QueryResponse(
            answer=self.answer,
            retrieved_chunk_ids=self.retrieved_chunk_ids,
            retrieved_sources=self.retrieved_sources,
            latency_seconds=self.latency_seconds,
        )
