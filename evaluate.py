"""
Module 7 Project — Semantic Search Engine
==========================================
evaluate.py — precision and recall for your search system

Run with:
    python evaluate.py
    python evaluate.py --n-results 5 --threshold 0.4

Define your evaluation set in EVAL_SET, then run this script against
different index configurations (different chunk sizes) to compare results.
"""

import argparse
from search import search

# Define your evaluation set here.
# Each entry needs a query and the source filenames you expect to be relevant.
# Use at least 5 queries for the chunking experiment.
EVAL_SET = [
    {
        "query": "How do vector embeddings represent semantic similarity?",
        "relevant_sources": ["embeddings-and-vectors.txt", "llms-and-ai.txt"],
    },
    {
        "query": "How do you build REST endpoints and handle status codes?",
        "relevant_sources": ["rest-apis.txt", "fastapi.txt"],
    },
    {
        "query": "How do SQL indexes and joins improve query performance?",
        "relevant_sources": ["sql-databases.txt"],
    },
    {
        "query": "What are Python decorators, generators, and context managers?",
        "relevant_sources": ["python-advanced.txt", "python-fundamentals.txt"],
    },
    {
        "query": "How do you create interactive widgets and state in Streamlit?",
        "relevant_sources": ["streamlit.txt"],
    },
]


def precision_recall(
    retrieved_sources: list[str], relevant_sources: list[str]
) -> tuple[float, float]:
    """
    Compute source-level precision and recall.

    Returns (precision, recall) as floats in [0, 1].
    """
    if not retrieved_sources:
        return 0.0, 0.0

    relevant_set = set(relevant_sources)
    if not relevant_set:
        return 0.0, 0.0

    retrieved_set = set(retrieved_sources)
    true_positives = len(retrieved_set & relevant_set)

    precision = true_positives / len(retrieved_set)
    recall = true_positives / len(relevant_set)
    return round(precision, 4), round(recall, 4)


def evaluate(n_results: int = 5, distance_threshold: float = None):
    """
    Run every query in EVAL_SET and print per-query and average precision/recall.
    """
    if not EVAL_SET:
        print("EVAL_SET is empty. Add queries before running evaluation.")
        return

    precisions: list[float] = []
    recalls: list[float] = []

    print(f"\nEvaluating {len(EVAL_SET)} queries (n_results={n_results}, threshold={distance_threshold})\n")
    print("-" * 80)

    for idx, item in enumerate(EVAL_SET, start=1):
        query = item["query"]
        relevant = item["relevant_sources"]
        results = search(
            query=query,
            n_results=n_results,
            distance_threshold=distance_threshold,
        )
        retrieved_sources = [r["source"] for r in results]
        prec, rec = precision_recall(retrieved_sources, relevant)
        precisions.append(prec)
        recalls.append(rec)

        print(f"Q{idx}: {query}")
        print(f"  Expected : {relevant}")
        print(f"  Retrieved: {list(dict.fromkeys(retrieved_sources))}")
        print(f"  Precision: {prec:.4f} | Recall: {rec:.4f}")
        print("-" * 80)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    print(f"\nAverage Precision: {avg_precision:.4f}")
    print(f"Average Recall   : {avg_recall:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate search quality")
    parser.add_argument("--n-results", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    evaluate(n_results=args.n_results, distance_threshold=args.threshold)