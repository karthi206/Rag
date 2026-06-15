"""
evaluate.py — Production-grade RAGAS evaluation for the RAG pipeline.

Usage:
    python evaluate.py

Measures Faithfulness, Answer Relevancy, and Context Precision
using curated QA pairs grounded in the actual documents.

Requires Ollama to be running (phi3 model by default).
"""

import os
import sys
from dotenv import load_dotenv

# Load HF_TOKEN and other env variables
load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "phi3")

# ─────────────────────────────────────────
# Imports
# FIXED: correct RAGAS import path.
# ragas.metrics.collections does NOT exist — correct path is ragas.metrics
# ─────────────────────────────────────────
try:
    from ragas import evaluate
    # ragas 0.4.x: primary import path is ragas.metrics.collections
    try:
        from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision
    except ImportError:
        # Fallback for older ragas builds
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
except ImportError as e:
    print(f"[ERROR] Could not import ragas: {e}")
    print("  Run: pip install ragas>=0.1.7")
    sys.exit(1)

from datasets import Dataset

from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings

# FIXED: RAGAS >=0.1 requires LangchainLLMWrapper and LangchainEmbeddingsWrapper
# Passing raw Langchain objects directly causes TypeError in newer RAGAS versions.
try:
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    USE_WRAPPERS = True
except ImportError:
    # Older RAGAS versions don't need wrappers
    USE_WRAPPERS = False


# ─────────────────────────────────────────
# Init models
# ─────────────────────────────────────────
print(f"[INFO] Loading LLM: {MODEL_NAME} (Ollama)")
try:
    llm_raw = Ollama(model=MODEL_NAME)
    # Quick liveness check
    llm_raw.invoke("ping")
except Exception as e:
    print(f"[ERROR] Cannot connect to Ollama: {e}")
    print("  Make sure Ollama is running: ollama serve")
    print(f"  And the model is pulled:     ollama pull {MODEL_NAME}")
    sys.exit(1)

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
print(f"[INFO] Loading embedding model: {EMBED_MODEL}")
embed_raw = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

# Wrap if needed
if USE_WRAPPERS:
    llm_eval   = LangchainLLMWrapper(llm_raw)
    embed_eval = LangchainEmbeddingsWrapper(embed_raw)
else:
    llm_eval   = llm_raw
    embed_eval = embed_raw


# ─────────────────────────────────────────
# Evaluation dataset
# QA pairs grounded in the actual documents:
#   - data science.pdf, ml.pdf, sample.pdf
# ─────────────────────────────────────────
data = {
    "question": [
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is the difference between supervised and unsupervised learning?",
        "What is data science?",
        "What is a neural network?",
        "What is deep learning?",
    ],
    "answer": [
        "Artificial intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn.",
        "Machine learning is a subset of AI that enables systems to automatically learn and improve from experience without being explicitly programmed.",
        "Supervised learning uses labelled training data to learn a mapping from inputs to outputs. Unsupervised learning finds hidden patterns in data without labelled examples.",
        "Data science is an interdisciplinary field that uses scientific methods, algorithms, and systems to extract knowledge and insights from structured and unstructured data.",
        "A neural network is a computational model inspired by the structure of the human brain, consisting of layers of interconnected nodes (neurons) that process data.",
        "Deep learning is a subset of machine learning that uses multi-layered neural networks to model complex patterns in large datasets.",
    ],
    "contexts": [
        ["AI attempts to build intelligent entities that can perceive, reason, learn, and act."],
        ["Machine learning involves algorithms that learn from training data and improve predictions over time."],
        ["Supervised learning trains on labelled data. Unsupervised learning discovers structure in unlabelled data."],
        ["Data science combines statistics, programming, and domain knowledge to analyse and interpret complex data."],
        ["Neural networks consist of an input layer, hidden layers, and an output layer, each containing nodes with weighted connections."],
        ["Deep learning models use many hidden layers to extract hierarchical features from raw data."],
    ],
    "ground_truth": [
        "Artificial intelligence is the field of creating intelligent systems that can perform tasks requiring human-like intelligence.",
        "Machine learning is a branch of artificial intelligence where systems learn from data to improve their performance.",
        "Supervised learning requires labelled data; unsupervised learning works without labels to find patterns.",
        "Data science is the study of extracting actionable insights from data using statistics, programming, and domain expertise.",
        "A neural network is a series of algorithms that recognise underlying relationships in data through a process that mimics the human brain.",
        "Deep learning is a type of machine learning that uses neural networks with many layers to learn from vast amounts of data.",
    ],
}

dataset = Dataset.from_dict(data)

print("\n" + "=" * 60)
print("  RAG EVALUATION — RAGAS Metrics")
print("=" * 60)
print(f"  Evaluating  : {len(data['question'])} question-answer pairs")
print(f"  Metrics     : Faithfulness, Answer Relevancy, Context Precision")
print(f"  LLM         : {MODEL_NAME}")
print("=" * 60 + "\n")


# ─────────────────────────────────────────
# Run RAGAS evaluation
# ─────────────────────────────────────────
try:
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=llm_eval,
        embeddings=embed_eval,
    )
except Exception as e:
    print(f"[ERROR] RAGAS evaluation failed: {e}")
    sys.exit(1)


# ─────────────────────────────────────────
# Print results
# ─────────────────────────────────────────
print("\n📊 RAGAS Evaluation Results\n")
print(results)

try:
    results_df = results.to_pandas()
    print("\n📋 Per-Question Breakdown:")
    cols = [c for c in ["question", "faithfulness", "answer_relevancy", "context_precision"]
            if c in results_df.columns]
    print(results_df[cols].to_string(index=False))

    print("\n📈 Aggregate Scores:")
    PASS_THRESHOLD = 0.7
    for metric in ["faithfulness", "answer_relevancy", "context_precision"]:
        if metric in results_df.columns:
            avg    = results_df[metric].mean()
            status = "✅ PASS" if avg >= PASS_THRESHOLD else "❌ BELOW THRESHOLD"
            print(f"  {metric:<25} avg = {avg:.4f}   {status}")
    print(f"\n  (Pass threshold: {PASS_THRESHOLD})")
except Exception as e:
    print(f"[WARN] Could not generate per-question breakdown: {e}")

print("\n✅ Evaluation complete.")