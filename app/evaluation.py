import time


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


def evaluate_rag(eval_data, ask_fn):

    total_keyword_score = 0
    total_keyword_possible = 0

    total_grounding = 0

    print("\n========================")
    print("🧪 ADVANCED RAG EVALUATION")
    print("========================")

    for i, test in enumerate(eval_data):

        query = test["query"]
        expected = test["expected_keywords"]

        print(f"\n🔍 Test {i+1}")
        print(f"Query: {query}")

        start = time.time()

        result = ask_fn(query)

        latency = time.time() - start

        answer = result["answer"]
        context = result["context"]

        keyword_score = keyword_match_score(
            answer,
            expected
        )

        grounding = grounding_score(
            answer,
            context
        )

        total_keyword_score += keyword_score
        total_keyword_possible += len(expected)

        total_grounding += grounding

        print(f"\n✅ Answer:")
        print(answer)

        print(f"\n📄 Retrieved Chunks:")
        print(result["retrieved_chunk_ids"])

        print(f"\n📚 Sources:")
        print(result["retrieved_sources"])

        print(f"\n📊 Keyword Score:")
        print(f"{keyword_score}/{len(expected)}")

        print(f"\n🧠 Grounding Score:")
        print(f"{grounding:.2f}")

        print(f"\n⚡ Latency:")
        print(f"{latency:.2f} sec")

        print("\n------------------------")

    final_keyword = (
        total_keyword_score / total_keyword_possible
    ) * 100

    avg_grounding = (
        total_grounding / len(eval_data)
    )

    print("\n========================")
    print("🏁 FINAL RESULTS")
    print("========================")

    print(f"\n🎯 Keyword Accuracy:")
    print(f"{final_keyword:.2f}%")

    print(f"\n🧠 Avg Grounding Score:")
    print(f"{avg_grounding:.2f}")

# Query Robustness Problem

# Production systems solve this with:

# spell correction
# query rewriting
# metadata filtering
# hybrid retrieval