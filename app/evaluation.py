import time
import re

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# embedding_model = SentenceTransformer(
#     "sentence-transformers/all-MiniLM-L6-v2"
# )


def keyword_match_score(answer, expected_keywords):
    matches = 0

    for keyword in expected_keywords:
        if keyword.lower() in answer.lower():
            matches += 1

    return matches


def grounding_score(answer, context):
    answer_words = answer.split()

    grounded = 0

    for word in answer_words:
        if word.lower() in context.lower():
            grounded += 1

    return grounded / max(len(answer_words), 1)

def precision_at_k(retrieved_ids, expected_ids, k):

    retrieved_top_k = retrieved_ids[:k]

    relevant = 0

    for chunk_id in retrieved_top_k:
        if chunk_id in expected_ids:
            relevant += 1

    return relevant / k


# def semantic_hallucination_detection(
#     answer,
#     context,
#     threshold=0.55
# ):

#     hallucinated_sentences = []

#     # Split answer into lines
#     sentences = [
#         s.strip()
#         for s in answer.split("\n")
#         if len(s.strip()) > 5
#     ]

#     # Embed context once
#     context_embedding = embedding_model.encode([context])

#     for sentence in sentences:

#         # Embed answer sentence
#         sentence_embedding = embedding_model.encode([sentence])

#         # Compute similarity
#         similarity = cosine_similarity(
#             sentence_embedding,
#             context_embedding
#         )[0][0]

#         # Low similarity → possible hallucination
#         if similarity < threshold:

#             hallucinated_sentences.append({
#                 "sentence": sentence,
#                 "similarity": round(float(similarity), 2)
#             })

#     return hallucinated_sentences

def evaluate_rag(eval_data, ask_fn):

    total_keyword_score = 0
    total_keyword_possible = 0

    total_grounding = 0
    
    total_precision = 0
    
    total_hallucinations = 0

    print("\n========================")
    print("🧪 ADVANCED RAG EVALUATION")
    print("========================")

    for i, test in enumerate(eval_data):

        query = test["query"]
        expected = test["expected_keywords"]
        expected_chunk_ids = test["expected_chunk_ids"]

        print(f"\n🔍 Test {i+1}")
        print(f"Query: {query}")

        start = time.time()

        result = ask_fn(query)

        latency = time.time() - start

        answer = result["answer"]
        context = result["context"]

        retrieved_chunk_ids = result["retrieved_chunk_ids"]

        precision = precision_at_k(
            retrieved_chunk_ids,
            expected_chunk_ids,
            k=1
        )
        
        keyword_score = keyword_match_score(
            answer,
            expected
        )

        grounding = grounding_score(
            answer,
            context
        )
        
        # hallucinations = semantic_hallucination_detection(
        #     answer,
        #     context
        # )

        total_precision += precision
        
        total_keyword_score += keyword_score
        total_keyword_possible += len(expected)

        total_grounding += grounding
        
        # total_hallucinations += len(hallucinations)

        print(f"\n✅ Answer:")
        print(answer)

        print(f"\n📄 Retrieved Chunks:")
        print(result["retrieved_chunk_ids"])

        print(f"\n📚 Sources:")
        print(result["retrieved_sources"])

        print(f"\n🎯 Precision@1:")
        print(f"{precision:.2f}")

        print(f"\n📊 Keyword Score:")
        print(f"{keyword_score}/{len(expected)}")

        print(f"\n🧠 Grounding Score:")
        print(f"{grounding:.2f}")
        
        # print(f"\n🚨 Hallucinations:")

        # if len(hallucinations) == 0:
        #     print("None")
        # else:
        #     for h in hallucinations:
        #         print(
        #             f"- {h['sentence']} "
        #             f"(similarity={h['similarity']})"
        #         )

        print(f"\n⚡ Latency:")
        print(f"{latency:.2f} sec")

        print("\n------------------------")

    final_keyword = (
        total_keyword_score / total_keyword_possible
    ) * 100

    avg_grounding = (
        total_grounding / len(eval_data)
    )
    
    avg_precision = total_precision / len(eval_data)

    print("\n========================")
    print("🏁 FINAL RESULTS")
    print("========================")
    
    print(f"\n🎯 Avg Precision@1:")
    print(f"{avg_precision:.2f}")

    print(f"\n🎯 Keyword Accuracy:")
    print(f"{final_keyword:.2f}%")

    print(f"\n🧠 Avg Grounding Score:")
    print(f"{avg_grounding:.2f}")
    
    # Embeddings semantic search is not sufficient alone for faithfulness evaluation, also tried string matching
    # print(f"\n🚨 Total Hallucinations:")
    # print(total_hallucinations)
    
    
# Query Robustness Problem

# Production systems solve this with:

# spell correction
# query rewriting
# metadata filtering
# hybrid retrieval