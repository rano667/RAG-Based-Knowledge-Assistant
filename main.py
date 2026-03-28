import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# # Loaded X pages ->  Text printed from PDF
# def load_documents():
#     loader = PyPDFLoader("data/invoice-0.pdf")
#     documents = loader.load()
#     return documents

# docs = load_documents()

# for i, doc in enumerate(docs[:3]):
#     print(f"\n--- Page {i} ---\n")
#     print(doc.page_content[:500])

def load_all_pdfs(folder_path):
    all_docs = []
    
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, file))
            all_docs.extend(loader.load())
    
    return all_docs

docs = load_all_pdfs("data")
print(f"Total pages loaded: {len(docs)}")

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700, # 500
        chunk_overlap=150 # 100
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks


chunks = split_documents(docs)

print(f"Total chunks created: {len(chunks)}")

for i, chunk in enumerate(chunks[:10]):
    print(f"\n--- Chunk {i} ---\n")
    print(chunk.page_content)