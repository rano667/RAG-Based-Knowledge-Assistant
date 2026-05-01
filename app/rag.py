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
    results = vectorstore.similarity_search(query, k=1)
    expanded = get_expanded_context(results, all_chunks)
    
    context = "\n\n".join([doc.page_content for doc in expanded])[:1500]
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer briefly using context only."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.3,
        max_tokens=100
    )
    
    return response.choices[0].message.content

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