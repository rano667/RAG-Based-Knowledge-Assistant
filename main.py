import os
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Step 1: Text extracted

# # Loaded X pages ->  Text printed from PDF
# def load_documents():
#     loader = PyPDFLoader("data/invoice-0.pdf")
#     documents = loader.load()
#     return documents

# docs = load_documents()

# for i, doc in enumerate(docs[:3]):
#     print(f"\n--- Page {i} ---\n")
#     print(doc.page_content[:500])

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
        chunk_size=700, # 500
        chunk_overlap=150 # 100
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks


chunks = split_documents(docs)

print(f"Total chunks created: {len(chunks)}")

# inspect chunks
# for i, chunk in enumerate(chunks[:10]):
#     print(f"\n--- Chunk {i} ---\n")
#     print(chunk.page_content)

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

# query 0
query = "What items are in invoice 0012820?"

results = vectorstore.similarity_search(query, k=3)

for i, res in enumerate(results):
    print(f"\n--- Result {i} ---\n")
    print(res.page_content)

# # query 1
# query1 = "What did Caitlin Roberts order?"

# results = vectorstore.similarity_search(query1, k=3)

# for i, res in enumerate(results):
#     print(f"\n--- Result {i} ---\n")
#     print(res.page_content)

# # query 2
# query2 = "What is the total due?"

# results = vectorstore.similarity_search(query2, k=3)

# for i, res in enumerate(results):
#     print(f"\n--- Result {i} ---\n")
#     print(res.page_content)