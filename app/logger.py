"""Application logging helpers."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any, Iterable

from langchain_core.documents import Document


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for container and log-collector friendly output."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event_data = getattr(record, "event_data", None)
        if event_data:
            payload.update(event_data)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure process logging once for the API or evaluation runner."""

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())
    if root_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter()
        if json_logs
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    )
    root_logger.addHandler(handler)


def log_event(logger: logging.Logger, event: str, **data: Any) -> None:
    """Log a structured application event."""

    logger.info(event, extra={"event_data": {"event": event, **data}})


def log_retrieval(
    logger: logging.Logger,
    query: str,
    documents: Iterable[Document],
) -> None:
    """Record retrieval metadata without dumping full document text."""

    docs = list(documents)
    log_event(
        logger,
        "rag_retrieval_completed",
        query=query,
        chunk_ids=[doc.metadata.get("chunk_id") for doc in docs],
        sources=[doc.metadata.get("source") for doc in docs],
    )
