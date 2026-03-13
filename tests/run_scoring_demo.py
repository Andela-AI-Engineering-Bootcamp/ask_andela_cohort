#!/usr/bin/env python3
"""
Demo: run scoring on sample answers and print accuracy (and other) results.

Use this to verify the scoring module and see how accuracy, specificity,
and conciseness are computed.

Usage (from project root):
    python tests/run_scoring_demo.py

To score your own answers, pass a JSON file with "answer" and "expected_points"
per question, or plug in answers from evaluate.py output.
"""
import json
import sys
from pathlib import Path

# Add project root so "rag" package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.scoring import (
    keyword_score,
    accuracy_score,
    conciseness_score,
    specificity_score,
    total_score,
)


def load_eval_set():
    """Load eval test set from tests/eval_test_set.json."""
    path = Path(__file__).parent / "eval_test_set.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    # Sample answers (good, partial, weak) to demonstrate accuracy differences
    sample_answers = [
        "RAG stands for Retrieval-Augmented Generation. We use a vector store and feed context documents into the LLM.",
        "RAG uses a vector store.",
        "AI is useful for many tasks.",
    ]
    eval_items = load_eval_set()
    # Use first question that has expected_points
    item = eval_items[0]
    question = item["question"]
    expected = item.get("expected_points", [])

    print("=" * 60)
    print("  Scoring demo — accuracy and other results")
    print("=" * 60)
    print(f"\nQuestion: {question}")
    print(f"Expected points: {expected}\n")

    rows = []
    for i, answer in enumerate(sample_answers, 1):
        kw = keyword_score(answer, expected)
        acc = accuracy_score(answer, expected)
        spec = specificity_score(answer)
        conc = conciseness_score(answer)
        tot = total_score(acc, spec, conc)
        rows.append({
            "answer_preview": answer[:50] + "..." if len(answer) > 50 else answer,
            "keyword_ratio": kw,
            "accuracy": acc,
            "specificity": spec,
            "conciseness": conc,
            "total": tot,
        })
        print(f"Answer {i}: {answer[:70]}{'...' if len(answer) > 70 else ''}")
        print(f"  keyword_score (0–1): {kw:.2f}  →  accuracy: {acc}  "
              f"specificity: {spec}  conciseness: {conc}  total: {tot}/9")
        print()

    print("-" * 60)
    print("Summary: accuracy = 3 (most keywords), 2 (some), 1 (few/none)")
    print("         total = accuracy + specificity + conciseness  (max 9)")
    print("=" * 60)


if __name__ == "__main__":
    main()
