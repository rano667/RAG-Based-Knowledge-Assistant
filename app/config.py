"""Runtime configuration for the RAG application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Configuration read from environment variables with local defaults."""

    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "data"))
    embedding_model: str = os.getenv(
        "RAG_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    generation_model: str = os.getenv("GROQ_GENERATION_MODEL", "llama-3.1-8b-instant")
    evaluator_model: str = os.getenv("GROQ_EVALUATOR_MODEL", "llama-3.1-8b-instant")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    retrieval_k: int = int(os.getenv("RAG_RETRIEVAL_K", "1"))
    max_context_chars: int = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "2500"))
    max_answer_tokens: int = int(os.getenv("RAG_MAX_ANSWER_TOKENS", "120"))
    generation_temperature: float = float(os.getenv("RAG_TEMPERATURE", "0.3"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    json_logs: bool = _as_bool(os.getenv("JSON_LOGS"))

    def __post_init__(self) -> None:
        """Reject configuration that would break splitting or retrieval."""

        if self.chunk_size <= 0:
            raise ValueError("RAG_CHUNK_SIZE must be greater than zero.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be between 0 and chunk size.")
        if self.retrieval_k <= 0:
            raise ValueError("RAG_RETRIEVAL_K must be greater than zero.")
        if self.max_context_chars <= 0 or self.max_answer_tokens <= 0:
            raise ValueError("RAG context and answer limits must be greater than zero.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-local settings instance."""

    return Settings()
