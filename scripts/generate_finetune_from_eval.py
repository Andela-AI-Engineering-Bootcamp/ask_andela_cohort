#!/usr/bin/env python3
"""
Generate fine-tuning records by running RAG on eval questions and saving (Q, A) as messages.

Reads tests/eval_questions_from_data.json, calls the RAG pipeline for each question,
and appends one JSONL record per question to data/finetune_dataset.jsonl (or --output).

Requires: ChromaDB populated (run python ingest.py first), OPENROUTER_API_KEY in .env.

Usage:
  python scripts/generate_finetune_from_eval.py
  python scripts/generate_finetune_from_eval.py --output data/finetune_from_eval.jsonl
  python scripts/generate_finetune_from_eval.py --limit 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import CHROMA_DIR, PROJECT_ROOT
from rag.vectorstore import load_vectorstore
from rag.pipeline import ask_andela


SYSTEM_PROMPT = (
    "You are Ask Andela, an AI study assistant for the A3 AI Engineering Bootcamp cohort. "
    "You answer student questions using only the course materials and channel discussions provided as context. "
    "Be concise (3–6 sentences), ground answers in the context, and reference course tools "
    "(ChromaDB, Gradio, QLoRA, OpenRouter) by name when relevant."
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate finetune JSONL from eval questions + RAG")
    parser.add_argument(
        "--questions",
        type=str,
        default=str(PROJECT_ROOT / "tests" / "eval_questions_from_data.json"),
        help="Path to JSON array of {question, ...} objects",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "finetune_dataset.jsonl"),
        help="Output JSONL path (default: append to data/finetune_dataset.jsonl)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=True,
        help="Append to output file if it exists (default: True)",
    )
    parser.add_argument(
        "--no-append",
        action="store_false",
        dest="append",
        help="Overwrite output file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of questions to process (for testing)",
    )
    args = parser.parse_args()

    questions_path = Path(args.questions)
    output_path = Path(args.output)

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        raise ValueError("Questions file must be a JSON array of objects with 'question' key")

    questions = []
    for item in items:
        if isinstance(item, dict) and "question" in item:
            questions.append(item["question"].strip())
        else:
            continue  # skip malformed entries

    if args.limit is not None:
        questions = questions[: args.limit]

    print(f"Loading vector store from {CHROMA_DIR} ...")
    collection = load_vectorstore(CHROMA_DIR)
    print(f"Running RAG on {len(questions)} questions ...")

    mode = "a" if args.append and output_path.exists() else "w"
    written = 0
    with open(output_path, mode, encoding="utf-8") as out:
        for i, q in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {q[:60]}...")
            result = ask_andela(q, collection)
            answer = result["answer"]
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": answer},
                ]
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} records to {output_path}")


if __name__ == "__main__":
    main()
