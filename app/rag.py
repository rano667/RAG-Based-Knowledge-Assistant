from multiprocessing import context
import time
from xmlrpc import client

from app.logger import (
    log_query,
    log_retrieval,
    log_context,
    log_response_time,
    log_final_answer
)

def get_expanded_context(results, all_chunks):
    expanded = []
    
    for res in results:
        idx = res.metadata["chunk_id"]
        
        expanded.append(all_chunks[idx])
        
        if idx - 1 >= 0:
            expanded.append(all_chunks[idx - 1])
        
        if idx + 1 < len(all_chunks):
            expanded.append(all_chunks[idx + 1])
    
    return expanded


def ask_rag(query, vectorstore, all_chunks, client):

    start_time = time.time()

    # Log query
    log_query(query)

    # Retrieval
    results = vectorstore.similarity_search(query, k=1)

    # Log retrieved chunks
    log_retrieval(results)

    # Expand context
    expanded = get_expanded_context(results, all_chunks)

    # Build context
    context = "\n\n".join(
        [doc.page_content for doc in expanded]
    )

    context = context[:2500]

    # Log context size
    log_context(context)
    # print("\n🧠 FULL CONTEXT:")
    # print(context)

    # LLM call
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    """
                    You are a document extraction system.

                    Rules:
                    1. ONLY use information explicitly present in the context.
                    2. DO NOT infer or complete missing items.
                    3. DO NOT guess.
                    4. If information is incomplete, say 'Information incomplete in retrieved context.'
                    5. Return concise bullet points only.
                    """
                )
            },
            {
                "role": "user",
                "content": (
                    f"""
                    Context:
                    {context}

                    Question:
                    {query}

                    Answer ONLY using exact information from context.
                    """
                )
            }
        ],
        temperature=0.3,
        max_tokens=120
    )

    answer = response.choices[0].message.content

    # Log latency
    log_response_time(start_time)

    # Log final answer
    log_final_answer(answer)

    retrieved_chunk_ids = [
    doc.metadata.get("chunk_id")
    for doc in results
    ]

    retrieved_sources = [
        doc.metadata.get("source")
        for doc in results
    ]

    return {
        "answer": answer,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_sources": retrieved_sources,
        "context": context
    }

# # ---- Tiny Llama RAG Logic ----
# def ask_rag(query, vectorstore, all_chunks, generator):
#     results = vectorstore.similarity_search(query, k=1)
    
#     expanded = get_expanded_context(results, all_chunks)
    
#     MAX_CONTEXT_CHARS = 1500
    
#     context = "\n\n".join([doc.page_content for doc in expanded])
#     context = context[:MAX_CONTEXT_CHARS]   # truncate
    
#     if len(context.strip()) == 0:
#         return "I don't know"
    
#     messages = [
#         {
#             "role": "system",
#             "content": "Answer ONLY from context. Keep it under 3 bullet points. If not found, say 'I don't know'."
#         },
#         {
#             "role": "user",
#             "content": f"Context:\n{context}\n\nQuestion: {query}"
#         }
#     ]
    
#     response = generator(
#         messages,
#         max_new_tokens=80,   # ↓ from 200
#         temperature=0.3
#     )
    
#     return response[0]["generated_text"][-1]["content"]