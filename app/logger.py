import time

def log_query(query):
    print("\n========================")
    print(f"🔍 QUERY: {query}")

def log_retrieval(results):
    print("\n📄 RETRIEVED CHUNKS:")
    
    for i, doc in enumerate(results):
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        
        print(f"\n--- Chunk {chunk_id} ---")
        print(doc.page_content[:300])

def log_context(context):
    print(f"\n🧠 CONTEXT LENGTH: {len(context)} chars")

def log_response_time(start_time):
    elapsed = time.time() - start_time
    print(f"\n⚡ LLM RESPONSE TIME: {elapsed:.2f} sec")

def log_final_answer(answer):
    print("\n✅ FINAL ANSWER:")
    print(answer)
    print("========================\n")