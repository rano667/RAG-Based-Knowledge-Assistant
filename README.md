# RAG-Based-Knowledge-Assistant

Production-grade Retrieval-Augmented Generation (RAG) system built using open-source embeddings, FAISS vector search, FastAPI, Streamlit, and LLM-based response generation.

This project processes invoice PDFs and enables users to query documents using natural language with grounded AI-generated responses.

---

# 📸 Demo

![sample screen image](images/example.png)

---

# 🚀 Features

- PDF document ingestion
- Semantic search using FAISS
- Open-source embeddings (`sentence-transformers`)
- Retrieval-Augmented Generation (RAG)
- Neighbor-aware retrieval for improved context continuity
- FastAPI backend
- Streamlit UI
- Modular production-style architecture
- Open-source / Groq-powered LLM support
- Startup lifecycle optimization
- Production-style folder structure

---

# 🧠 Problem Statement

Traditional keyword-based document search systems struggle to understand semantic meaning and contextual relationships inside documents.

This project solves that problem using Retrieval-Augmented Generation (RAG), allowing users to:

- Query invoice PDFs using natural language
- Retrieve semantically relevant information
- Generate grounded responses using LLMs
- Reduce hallucinations through retrieval-based context injection

---

# 🏗️ System Architecture

```text
User Query
    ↓
FastAPI Backend
    ↓
FAISS Semantic Search
    ↓
Relevant Chunk Retrieval
    ↓
Neighbor Chunk Expansion
    ↓
LLM Context Injection
    ↓
Groq / TinyLlama Response Generation
    ↓
Final Grounded Answer
```

---

# ⚙️ Tech Stack

## Backend

- Python
- FastAPI
- Uvicorn

---

## AI / RAG

- LangChain
- FAISS
- Sentence Transformers
- HuggingFace
- TinyLlama
- Groq API

---

## Frontend

- Streamlit

---

## Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

---

## LLMs Used

### Local Model

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Hosted Inference

```text
llama-3.1-8b-instant
```

---

# 📂 Project Structure

```text
rag-system/
│
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── rag.py               # Core RAG pipeline
│   ├── retriever.py         # Chunking + embeddings + FAISS
│   ├── llm.py               # LLM integration
│   ├── evaluation.py        # Evaluation logic
│   ├── logger.py            # Logging utilities
│
├── data/                    # PDF invoices
│
├── ui/
│   └── streamlit_app.py     # Frontend UI
│
├── images/
│
├── requirements.txt
└── README.md
```

---

# 🔍 How Retrieval Works

## Step 1 — PDF Ingestion

Invoice PDFs are loaded using:

```python
PyPDFLoader
```

---

## Step 2 — Chunking

Documents are split into semantic chunks using:

```python
RecursiveCharacterTextSplitter
```

Current configuration:

```python
chunk_size=700
chunk_overlap=150
```

---

## Step 3 — Embeddings

Chunks are converted into dense vector embeddings using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

---

## Step 4 — Vector Storage

Embeddings are stored inside:

```text
FAISS Vector Database
```

---

## Step 5 — Semantic Retrieval

When a user asks a query:

- Query is converted to embeddings
- FAISS retrieves semantically similar chunks
- Neighbor chunks are expanded for continuity

---

## Step 6 — LLM Generation

Retrieved context is passed into:

- TinyLlama (local)
- OR Groq-hosted Llama 3.1

The LLM generates grounded responses using retrieved context.

---

# 🧠 Key Engineering Decisions

## Neighbor-Aware Retrieval

### Problem

Semantic chunking caused invoice tables to split across chunks.

Example:

- Half the invoice items in chunk A
- Remaining items in chunk B

This caused incomplete answers.

---

### Solution

Implemented neighbor-aware retrieval:

- Retrieve top semantic chunk
- Expand previous + next chunks
- Preserve semantic continuity

This significantly improved retrieval quality.

---

## Startup Optimization

### Problem

Embeddings and vector store were reloading on every API request.

This caused:

- High latency
- Slow startup
- Poor user experience

---

### Solution

Used FastAPI lifespan startup events to preload:

- Documents
- Chunks
- Embeddings
- Vector store
- LLM

Result:

- Faster API response
- Reduced repeated computation

---

## Latency Optimization

### Problem

Local TinyLlama inference was slow on limited hardware.

System specs:

- Intel i3 CPU
- 8GB RAM

Inference caused:

- CPU saturation
- RAM exhaustion
- Slow responses

---

### Solution

Integrated Groq-hosted inference for:

- Faster generation
- Reduced local resource usage
- Better scalability

---

# 📌 Example Query
<!-- query = "What did Caitlin Roberts order?" -->
<!-- query = "What is the total due?" -->

```python
query = "What items are in invoice 0012820?"
```

---

# ✅ Example Response

```text
Invoice 0012820 includes the following items:

1. 10-700 - Exterior Protection (10)
2. 1-515 - Temporary Lighting (29)
3. 11-060 - Theater and Stage Equipment (17)
4. 1-600 - Product Requirements (Scope of Work) (20)
5. 12-050 - Fabrics (23)
6. 2-823 - PVC Fences and Gates (27)
7. 6-400 - Architectural Woodwork (26)
8. 2-820 - Fences and Gates (15)
9. 9-700 - Wall Finishes (1)
10. 2-795 - Porous Paving (30)
```

---

# 🔐 Environment Setup

## 1. Clone Repository

```bash
git clone <your_repo_url>
cd rag-system
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Create Groq API Key

Create a free API key from:

```text
https://console.groq.com
```

---

## 4. Create `.env` File

In the root directory:

```env
GROQ_API_KEY=your_api_key_here
```

---

## 5. Add `.env` to `.gitignore`

```bash
.env
```

---

# ▶️ Running the Project

## Start FastAPI Backend

```bash
uvicorn app.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

---

## Start Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

---

# 🌐 API Endpoint

## POST `/ask`

### Request

```json
{
  "query": "What items are in invoice 0012820?"
}
```

---

### Response

```json
{
  "answer": "Invoice 0012820 includes..."
}
```

---

# 🧪 Evaluation Strategy

The project includes a basic evaluation framework for:

- Retrieval quality
- Answer correctness
- Context grounding

Planned evaluation improvements:

- Precision@K
- Retrieval relevance scoring
- Hallucination detection
- Automated benchmark datasets

---

# 📊 Logging Strategy

The system is designed to log:

- User query
- Retrieved chunks
- Context size
- LLM latency
- Response generation time

This helps debug:

- Retrieval issues
- Hallucinations
- Performance bottlenecks

---

# 🚀 Future Improvements

## Retrieval

- Metadata filtering
- Hybrid search (BM25 + vector search)
- Table-aware chunking
- Better semantic splitting

---

## AI

- Multi-agent workflows
- Tool calling
- Agentic retrieval
- Evaluation agents

---

## Infra

- Dockerization
- AWS deployment
- ECS / ECR pipelines
- CI/CD
- GPU inference

---

## Product

- Authentication
- Multi-user support
- Chat history
- File upload UI
- Persistent vector DB

---

# 📚 Learning Outcomes

This project helped explore:

- Retrieval-Augmented Generation (RAG)
- Semantic search pipelines
- Vector databases
- Open-source embeddings
- Prompt engineering
- Context management
- AI backend architecture
- Latency optimization
- Production-style AI system design

---

# 👨‍💻 Author

## Ranjan Mondal

### Links

- GitHub: https://github.com/rano667
- LinkedIn: https://www.linkedin.com/in/ranjanmondal/

---

# ⭐ Final Note

This project was built not just as a tutorial implementation, but as a step toward production-grade AI systems engineering with strong focus on:

- System thinking
- Retrieval quality
- Modularity
- Optimization
- Real-world AI architecture

