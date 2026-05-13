def keyword_match_score(answer, expected_keywords):
    matches = 0
    
    for keyword in expected_keywords:
        if keyword.lower() in answer.lower():
            matches += 1
    
    return matches


def evaluate_rag(eval_data, ask_fn):
    
    total_score = 0
    total_possible = 0

    print("\n========================")
    print("🧪 STARTING EVALUATION")
    print("========================")

    for i, test in enumerate(eval_data):

        query = test["query"]
        expected = test["expected_keywords"]

        print(f"\n🔍 Test {i+1}")
        print(f"Query: {query}")

        answer = ask_fn(query)

        score = keyword_match_score(answer, expected)

        total_score += score
        total_possible += len(expected)

        print(f"\n✅ Answer:")
        print(answer)

        print(f"\n🎯 Expected Keywords:")
        print(expected)

        print(f"\n📊 Score: {score}/{len(expected)}")
        print("\n------------------------")

    final_score = (total_score / total_possible) * 100

    print("\n========================")
    print(f"🏁 FINAL SCORE: {final_score:.2f}%")
    print("========================")


# Real production systems later use:

# RAGAS
# DeepEval
# TruLens
# LLM judges