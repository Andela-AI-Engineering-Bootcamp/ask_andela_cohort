#!/usr/bin/env python3
"""
Ask Andela.

Build the vector store first:  python ingest.py
Run:                           python app.py [--share]
"""
import sys
import argparse

from rag.config import CHROMA_DIR
from rag.vectorstore import load_vectorstore
from ui.gradio.interface import build, STYLES_CSS

try:
    collection = load_vectorstore(CHROMA_DIR)
except Exception:
    sys.exit(
        "\nERROR: Vector store not found.\n"
        "Run `python ingest.py` first, then restart the app.\n"
    )

demo = build(collection)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Public Gradio link")
    args = parser.parse_args()
    demo.launch(share=args.share, css=STYLES_CSS)
