#!/usr/bin/env python3
"""
Evaluation runner - compares Baseline vs RAG on 12 held-out questions.

Outputs eval_results.json at the project root with columns for all three
comparison tiers (baseline / RAG / fine-tuned+RAG).

Usage:
    python scripts/evaluate.py                  # baseline + RAG
    python scripts/evaluate.py --no-baseline    # RAG only (saves API calls)
"""
import sys
import json
import argparse
from pathlib import Path


from rag.config import CHROMA_DIR, PROJECT_ROOT
from rag.vectorstore import load_vectorstore
from rag.pipeline import ask_andela
from rag.llm import baseline_answer


# ── Held-out evaluation set ───────────────────────────────────────────────────
# These questions must NOT appear in the fine-tuning dataset.
EVAL_QUESTIONS = [
    # RAG & vector stores
    {"id": "eval_01", "question": "What is RAG and why is it used in this course instead of just prompting the LLM directly?"},
    {"id": "eval_02", "question": "How does ChromaDB store and retrieve embeddings?"},
    {"id": "eval_03", "question": "What embedding model do we use and why was it chosen?"},
    # Fine-tuning
    {"id": "eval_04", "question": "What is QLoRA and why do we use it instead of full fine-tuning?"},
    {"id": "eval_05", "question": "Which base model should I fine-tune on Google Colab and why?"},
    # Tools
    {"id": "eval_06", "question": "How do I set up the fine-tuning environment on Colab?"},
    {"id": "eval_07", "question": "What is the role of Gradio in the final demo?"},
    # Programme logistics
    {"id": "eval_08", "question": "What are the deliverables for the capstone project and how is it assessed?"},
    {"id": "eval_09", "question": "When are behavioral assignments due each week?"},
    # Setup / troubleshooting
    {"id": "eval_10", "question": "How do I fix the 'openai module not found' error in VS Code?"},
    {"id": "eval_11", "question": "How do I install VS Build Tools on Windows?"},
    # Programme norms
    {"id": "eval_12", "question": "What is the difference between peer review and the capstone assessment?"},
]


# ── Runner ────────────────────────────────────────────────────────────────────
def run_evaluation(collection, *, include_baseline: bool = True) -> list[dict]:
    total   = len(EVAL_QUESTIONS)
    results = []

    for i, item in enumerate(EVAL_QUESTIONS, 1):
        print(f"[{i:02d}/{total}] {item['id']}: {item['question'][:65]}...")

        entry = {
            "id":       item["id"],
            "question": item["question"],
            # Answers (populated below)
            "baseline_answer":      None,
            "rag_answer":           None,
            "rag_sources":          None,
            "finetuned_rag_answer": None,   # filled in after QLoRA integration
            # Scores - fill manually after the demo (1-3 scale)
            "baseline_accuracy":         None,
            "baseline_specificity":      None,
            "baseline_conciseness":      None,
            "rag_accuracy":              None,
            "rag_specificity":           None,
            "rag_conciseness":           None,
            "finetuned_rag_accuracy":    None,
            "finetuned_rag_specificity": None,
            "finetuned_rag_conciseness": None,
        }

        if include_baseline:
            entry["baseline_answer"] = baseline_answer(item["question"])

        rag = ask_andela(item["question"], collection)
        entry["rag_answer"]  = rag["answer"]
        entry["rag_sources"] = rag["sources"]

        results.append(entry)

    return results


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ask Andela evaluation")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip no-context baseline calls (saves ~12 API calls)",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Ask Andela - Evaluation Runner")
    print("=" * 50)

    collection = load_vectorstore(CHROMA_DIR)
    results    = run_evaluation(collection, include_baseline=not args.no_baseline)

    output_path = PROJECT_ROOT / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    print(
        "\nNext steps:\n"
        "  1. Open eval_results.json and score each response 1-3 on "
        "Accuracy / Specificity / Conciseness.\n"
        "  2. After QLoRA fine-tuning, populate the 'finetuned_rag_answer' column "
        "and re-score.\n"
        "  3. Present the three-column comparison table in the demo."
    )


if __name__ == "__main__":
    main()
