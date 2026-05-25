# RAG-Based Knowledge Assistant

A document question-answering system for invoice PDFs using FastAPI, FAISS,
HuggingFace embeddings, Groq-hosted Llama 3.1 generation, Streamlit, custom
offline metrics, and optional RAGAS evaluation.

![sample screen image](images/example.png)

## Features

- PDF ingestion with stable chunk IDs for benchmarkable retrieval.
- Semantic retrieval over an in-memory FAISS index.
- Neighbor-aware context expansion for chunked invoice tables.
- Grounded Groq answer generation behind a typed RAG pipeline.
- FastAPI lifespan startup so heavy index/model clients load once per process.
- Streamlit UI for interactive queries.
- Structured logging with optional JSON output.
- Offline evaluation for Precision@1, keyword recall, grounding overlap,
  latency, experimental hallucination flags, and RAGAS metrics.

## Architecture

```text
app/
|-- config.py              # Environment-backed runtime settings
|-- llm.py                 # Groq generation boundary and prompt
|-- logger.py              # Text or JSON structured logging
|-- main.py                # FastAPI online entry point
|-- rag.py                 # Pipeline orchestration and context expansion
|-- retriever.py           # PDF ingestion, chunking, embeddings, FAISS
|-- schemas.py             # API and internal response shapes
`-- evaluation/
    |-- benchmark.py       # Offline benchmark dataset
    |-- metrics.py         # Custom metrics
    |-- ragas_eval.py      # Optional RAGAS adapter
    |-- report.py          # Evaluation result models and reporting
    `-- run_eval.py        # Offline CLI runner
```

The online API does not import or execute evaluation code. Both the API and the
offline runner build the same `RagPipeline`, so evaluation exercises the real
retrieval and generation path without coupling benchmark work to FastAPI.

## Runtime Flow

```text
PDF pages -> chunks -> HuggingFace embeddings -> FAISS
query -> top-k retrieval -> neighbor expansion -> bounded context
context + query -> Groq generation -> typed answer response
```

The public `/ask` response includes the answer, retrieval IDs/sources, and
latency. Full context remains internal so evaluation can score grounding without
returning retrieved document text to every API caller.

## Live Deployment

- Backend API: deployed on AWS ECS and available at
  `http://40.192.14.146:8000/ask`.
- Streamlit frontend: deployed on Streamlit Community Cloud at
  `https://rag-based-knowledge-assistant-667.streamlit.app/`.

## Setup

1. Create a virtual environment and install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

2. Create `.env` from `.env.example` and set `GROQ_API_KEY`.

3. Put PDF files in `data/` or set `RAG_DATA_DIR`.

## Run

Start the backend:

```bash
uvicorn app.main:app --reload
```

The deployed backend is available at:

```text
http://40.192.14.146:8000/ask
```

Build and run the API container:

```bash
docker build -t rag-knowledge-assistant .
docker run --rm -p 8000:8000 --env-file .env rag-knowledge-assistant
```

The Docker image uses the smaller `requirements-api.txt` runtime set. The root
`requirements.txt` keeps Streamlit and offline evaluation dependencies for local
development.

Start the UI in another terminal:

```bash
streamlit run ui/streamlit_app.py
```

The deployed Streamlit app is available at:

```text
https://rag-based-knowledge-assistant-667.streamlit.app/
```

Health check:

```text
GET http://127.0.0.1:8000/health
```

Ask endpoint:

```http
POST /ask
Content-Type: application/json

{"query": "What items are in invoice 0012820?"}
```

## Offline Evaluation

Custom benchmark metrics:

```bash
python -m app.evaluation.run_eval
```

Custom metrics plus RAGAS:

```bash
python -m app.evaluation.run_eval --ragas
```

The RAGAS path uses the same generated answers and retrieved context produced by
the offline pipeline. It preserves faithfulness, answer relevancy, context
precision, and context recall with ChatGroq as evaluator and HuggingFace
embeddings.

## Configuration

Defaults live in `.env.example`. Important deployment knobs are:

- `RAG_DATA_DIR` for the mounted PDF directory.
- `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, and `RAG_RETRIEVAL_K`.
- `RAG_MAX_CONTEXT_CHARS` and `RAG_MAX_ANSWER_TOKENS`.
- `GROQ_GENERATION_MODEL` and `GROQ_EVALUATOR_MODEL`.
- `LOG_LEVEL` and `JSON_LOGS`.

## Engineering Notes

- PDF ordering is sorted before chunk IDs are assigned so benchmark chunk IDs
  stay stable for the same dataset.
- Custom word-overlap grounding is a cheap review signal. RAGAS faithfulness is
  the stronger LLM-based check when evaluator cost and latency are acceptable.
- The FAISS index is process-local and rebuilt at startup. Persisted indexes,
  file upload ingestion, table-aware parsing, and hybrid search are natural next
  steps for a larger deployment.
