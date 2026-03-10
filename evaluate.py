from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
from datasets import Dataset

data = {
    "question": [
        "What is AI?",
        "What is machine learning?"
    ],
    "answer": [
        "Artificial intelligence is the field of building intelligent systems.",
        "Machine learning is a subset of AI where systems learn from data."
    ],
    "contexts": [
        ["AI attempts to build intelligent entities."],
        ["Machine learning involves algorithms that learn from data."]
    ]
}

dataset = Dataset.from_dict(data)

results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_relevancy
    ]
)

print(results)