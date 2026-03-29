from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel

from app.retriever import split_documents, create_vector_store
from app.llm import load_llm
from app.rag import ask_rag

from langchain_community.document_loaders import PyPDFLoader
import os

# ---- Global Variables ----
vectorstore = None
chunks = None
generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, chunks, generator
    
    print("🚀 Starting up... Loading models and data")
    
    documents = load_documents("data")
    print(f"Loaded docs: {len(documents)}")
    
    chunks = split_documents(documents)
    print(f"Chunks created: {len(chunks)}")
    
    vectorstore, _ = create_vector_store(chunks)
    print("Vector store ready")
    
    generator = load_llm()
    print("LLM loaded")
    
    yield
    
    print("🛑 Shutting down")

# ---- INIT ----
app = FastAPI(lifespan=lifespan)

# Load everything once
def load_documents(folder_path="data"):
    all_docs = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            all_docs.extend(loader.load())
    
    return all_docs

# (later we optimize startup)
# documents = load_documents("data")
# print(f"Loaded documents: {len(documents)}")
# chunks = split_documents(documents)
# print(f"Total chunks: {len(chunks)}")

# vectorstore, embeddings = create_vector_store(chunks)
# generator = load_llm()

# ---- API ----
class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask(request: QueryRequest):
    answer = ask_rag(request.query, vectorstore, chunks, generator)
    return {"answer": answer}