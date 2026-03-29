import os
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline

# Step 1: Text extracted

# Scalable
def load_all_pdfs(folder_path):
    all_docs = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            all_docs.extend(loader.load())
    
    return all_docs

docs = load_all_pdfs("data")
print(f"Total pages loaded: {len(docs)}")

# Step 2: Chunking

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add index metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    
    return chunks


chunks = split_documents(docs)

print(f"Total chunks created: {len(chunks)}")

def get_expanded_context(results, all_chunks):
    expanded_chunks = []
    
    for res in results:
        idx = res.metadata["chunk_id"]
        
        # current
        expanded_chunks.append(all_chunks[idx])
        
        # previous
        if idx - 1 >= 0:
            expanded_chunks.append(all_chunks[idx - 1])
        
        # next
        if idx + 1 < len(all_chunks):
            expanded_chunks.append(all_chunks[idx + 1])
    
    return expanded_chunks

# Step 3: Create Embeddings + Vector Store

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )
    
    return vectorstore


vectorstore = create_vector_store(chunks)

print("Vector store created successfully")

# CONNECT RAG + TinyLlama (FULL SYSTEM)

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device_map="auto"
)

def ask_rag(query, vectorstore, all_chunks):
    results = vectorstore.similarity_search(query, k=1)  # top 1
    
    expanded = get_expanded_context(results, all_chunks)
    
    context = "\n\n".join([doc.page_content for doc in expanded])
    
    messages = [
        {
            "role": "system",
            "content": "Answer ONLY from context. If missing, say 'I don't know'."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]
    
    response = generator(messages, max_new_tokens=200, temperature=0.3)
    
    return response[0]["generated_text"][-1]["content"]

query = "What items are in invoice 0012820?"
# query = "What did Caitlin Roberts order?"
# query = "What is the total due?"

answer = ask_rag(query, vectorstore, chunks)

print("\n=== FINAL ANSWER ===\n")
print(answer)