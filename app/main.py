"""FastAPI entry point for online RAG requests."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, cast

from fastapi import FastAPI, HTTPException, Request

from app.config import get_settings
from app.logger import configure_logging, log_event
from app.rag import RagPipeline, build_rag_pipeline
from app.schemas import QueryRequest, QueryResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize heavy runtime dependencies once per API process."""

    settings = get_settings()
    configure_logging(settings.log_level, settings.json_logs)
    log_event(logger, "api_startup_started")
    app.state.rag_pipeline = build_rag_pipeline(settings)
    log_event(logger, "api_startup_completed")
    yield
    log_event(logger, "api_shutdown")


app = FastAPI(
    title="RAG Knowledge Assistant",
    version="1.0.0",
    lifespan=lifespan,
)


def get_pipeline(request: Request) -> RagPipeline:
    """Read the initialized RAG pipeline from FastAPI application state."""

    return cast(RagPipeline, request.app.state.rag_pipeline)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight liveness response."""

    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
def ask(request_body: QueryRequest, request: Request) -> QueryResponse:
    """Answer a question from documents loaded during API startup."""

    query = request_body.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="query must not be blank")

    result = get_pipeline(request).ask(query)
    return result.to_api_response()
