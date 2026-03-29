from fastapi import FastAPI
from pydantic import BaseModel

from app.retriever import split_documents, create_vector_store
from app.llm import load_llm
from app.rag import ask_rag

from langchain_community.document_loaders import PyPDFLoader
import os

# ---- INIT ----
app = FastAPI()

# Load everything once
def load_documents(folder_path="data"):
    all_docs = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            all_docs.extend(loader.load())
    
    return all_docs
# (later we optimize startup)

documents = load_documents("data")
print(f"Loaded documents: {len(documents)}")
chunks = split_documents(documents)
print(f"Total chunks: {len(chunks)}")

vectorstore, embeddings = create_vector_store(chunks)
generator = load_llm()

# ---- API ----
class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask(request: QueryRequest):
    answer = ask_rag(request.query, vectorstore, chunks, generator)
    return {"answer": answer}