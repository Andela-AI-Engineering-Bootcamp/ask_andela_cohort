#!/usr/bin/env python3
"""
Gradio UI for Ask Andela.

Requires the vector store to be built first:
    python ingest.py

Usage:
    python app.py
    python app.py --share      # public Gradio link for demo
"""
import sys
import argparse
from pathlib import Path


import gradio as gr

from rag.config import CHROMA_DIR
from rag.vectorstore import load_vectorstore
from rag.pipeline import ask_andela


# ── Load vector store once at startup ────────────────────────────────────────
try:
    collection = load_vectorstore(CHROMA_DIR)
except Exception:
    print(
        "\nERROR: Vector store not found.\n"
        "Run `python scripts/ingest.py` first, then restart the app.\n"
    )
    sys.exit(1)


# ── Gradio callback ───────────────────────────────────────────────────────────
def query(question: str) -> tuple[str, str]:
    if not question or not question.strip():
        return "Please enter a question.", ""
    try:
        result = ask_andela(question.strip(), collection)
        sources_md = "**Sources cited:**\n" + "\n".join(
            f"• {src}" for src in result["sources"]
        )
        return result["answer"], sources_md
    except Exception as exc:
        return f"Something went wrong: {exc}", ""


# ── UI definition ─────────────────────────────────────────────────────────────
EXAMPLES = [
    ["What is RAG and why do we use it in this course?"],
    ["What is QLoRA and why is it used for fine-tuning?"],
    ["What are the submission requirements for the capstone project?"],
    ["How does ChromaDB store and retrieve embeddings?"],
    ["What model should I use for fine-tuning on Google Colab?"],
    ["How do I install VS Build Tools on Windows?"],
    ["When are behavioral assignments due?"],
    ["What is the format of submitting daily squad updates?"],
]

with gr.Blocks(title="Ask Andela") as demo:

    gr.Markdown(
        """
        # Ask Andela
        **Your AI study assistant for the A3 AI Engineering Bootcamp.**
        Answers are grounded in the cohort's own course materials and channel discussions.
        """
    )

    question_box = gr.Textbox(
        label="Your question",
        placeholder="e.g. What is the difference between RAG and fine-tuning?",
        lines=2,
    )
    submit_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        with gr.Column(scale=3):
            answer_box = gr.Textbox(label="Answer", lines=7, interactive=False)
        with gr.Column(scale=1):
            sources_box = gr.Markdown()

    gr.Examples(examples=EXAMPLES, inputs=question_box, label="Example questions")

    submit_btn.click(fn=query, inputs=question_box, outputs=[answer_box, sources_box])
    question_box.submit(fn=query, inputs=question_box, outputs=[answer_box, sources_box])

    gr.Markdown(
        "---\n"
        "*Powered by [OpenRouter](https://openrouter.ai) · "
        "Embeddings: all-MiniLM-L6-v2 · Vector store: ChromaDB*"
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    demo.launch(share=args.share)
