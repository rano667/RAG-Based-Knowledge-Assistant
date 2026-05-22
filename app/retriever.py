"""Document ingestion, chunking, and vector retrieval helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_documents(folder_path: Path | str) -> list[Document]:
    """Load all PDF pages from a directory in stable filename order."""

    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"PDF data directory does not exist: {folder}")

    documents: list[Document] = []
    for pdf_path in sorted(folder.glob("*.pdf")):
        documents.extend(PyPDFLoader(str(pdf_path)).load())
    return documents


def split_documents(
    documents: Sequence[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split PDF pages and attach stable chunk IDs used by evaluation."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks = splitter.split_documents(documents)
    
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = chunk.metadata.get("source", "unknown")
    
    return chunks


def create_vector_store(
    chunks: Sequence[Document],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> tuple[FAISS, HuggingFaceEmbeddings]:
    """Embed chunks into an in-memory FAISS vector store."""

    if not chunks:
        raise ValueError("Cannot create a vector store without document chunks.")

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return vectorstore, embeddings


def retrieve_chunks(vectorstore: FAISS, query: str, k: int) -> list[Document]:
    """Return the most similar chunks for a user query."""

    return vectorstore.similarity_search(query, k=k)
