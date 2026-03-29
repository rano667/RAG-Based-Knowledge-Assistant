import time

def evaluate_rag(eval_data, ask_fn):
    results = []

    for test in eval_data:
        query = test["query"]
        expected = test["expected_keywords"]

        start = time.time()
        answer = ask_fn(query)
        latency = time.time() - start

        score = sum([1 for word in expected if word.lower() in answer.lower()])

        results.append({
            "query": query,
            "answer": answer,
            "score": score,
            "latency": latency
        })

        print("\n====================")
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        print(f"Score: {score}/{len(expected)}")
        print(f"Latency: {latency:.2f}s")

    return results