from datasets import Dataset

from ragas import evaluate
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

evaluator_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def run_ragas_evaluation(
    question,
    answer,
    contexts,
    ground_truth
):

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth]
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=evaluator_llm,
        embeddings=embeddings
    )

    return result