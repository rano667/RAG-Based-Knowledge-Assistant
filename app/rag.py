"""Core RAG orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import Settings, get_settings
from app.llm import generate_grounded_answer, load_llm
from app.logger import log_event, log_retrieval
from app.retriever import (
    create_vector_store,
    load_pdf_documents,
    retrieve_chunks,
    split_documents,
)
from app.schemas import RagResponse

logger = logging.getLogger(__name__)


def expand_neighbor_context(
    retrieved_chunks: Sequence[Document],
    all_chunks: Sequence[Document],
) -> list[Document]:
    """Return retrieved chunks with immediate neighbors, without duplicates."""

    expanded: list[Document] = []
    seen_chunk_ids: set[int] = set()

    for retrieved in retrieved_chunks:
        chunk_id = retrieved.metadata.get("chunk_id")
        if not isinstance(chunk_id, int):
            continue

        for neighbor_id in (chunk_id - 1, chunk_id, chunk_id + 1):
            if neighbor_id < 0 or neighbor_id >= len(all_chunks):
                continue
            if neighbor_id in seen_chunk_ids:
                continue
            expanded.append(all_chunks[neighbor_id])
            seen_chunk_ids.add(neighbor_id)

    return expanded


def build_context(documents: Sequence[Document], max_chars: int) -> str:
    """Join retrieved document content with a defensive character limit."""

    return "\n\n".join(doc.page_content for doc in documents)[:max_chars]


@dataclass
class RagPipeline:
    """Stateful RAG service built once at API startup or offline evaluation."""

    vectorstore: FAISS
    chunks: list[Document]
    llm_client: Groq
    settings: Settings

    def ask(self, query: str) -> RagResponse:
        """Retrieve document context and generate one grounded answer."""

        started_at = perf_counter()
        retrieved = retrieve_chunks(
            self.vectorstore,
            query=query,
            k=self.settings.retrieval_k,
        )
        log_retrieval(logger, query, retrieved)

        expanded_context = expand_neighbor_context(retrieved, self.chunks)
        context = build_context(expanded_context, self.settings.max_context_chars)
        answer = generate_grounded_answer(
            self.llm_client,
            query=query,
            context=context,
            model=self.settings.generation_model,
            temperature=self.settings.generation_temperature,
            max_tokens=self.settings.max_answer_tokens,
        )
        latency_seconds = perf_counter() - started_at

        log_event(
            logger,
            "rag_answer_generated",
            context_chars=len(context),
            latency_seconds=round(latency_seconds, 4),
        )
        return RagResponse(
            answer=answer,
            retrieved_chunk_ids=[doc.metadata.get("chunk_id") for doc in retrieved],
            retrieved_sources=[doc.metadata.get("source") for doc in retrieved],
            context=context,
            latency_seconds=latency_seconds,
        )


def build_rag_pipeline(settings: Settings | None = None) -> RagPipeline:
    """Load PDF data, vector index, and Groq client once."""

    active_settings = settings or get_settings()
    documents = load_pdf_documents(active_settings.data_dir)
    chunks = split_documents(
        documents,
        chunk_size=active_settings.chunk_size,
        chunk_overlap=active_settings.chunk_overlap,
    )
    vectorstore, _ = create_vector_store(chunks, active_settings.embedding_model)
    log_event(
        logger,
        "rag_pipeline_built",
        documents=len(documents),
        chunks=len(chunks),
        data_dir=str(active_settings.data_dir),
    )
    return RagPipeline(
        vectorstore=vectorstore,
        chunks=chunks,
        llm_client=load_llm(),
        settings=active_settings,
    )


def ask_rag(
    query: str,
    vectorstore: FAISS,
    all_chunks: list[Document],
    client: Groq,
    settings: Settings | None = None,
) -> RagResponse:
    """Backward-compatible functional entry point for the RAG pipeline."""

    return RagPipeline(
        vectorstore=vectorstore,
        chunks=all_chunks,
        llm_client=client,
        settings=settings or get_settings(),
    ).ask(query)
