#!/usr/bin/env python3
"""
Ingestion pipeline — build (or rebuild) the ChromaDB vector store.

Run this once before starting the app, and again whenever the data/ folder changes.

Usage:
    python scripts/ingest.py
"""
import sys
from pathlib import Path

# Allow imports from scripts/allscripts/ regardless of CWD

from rag.config import DATA_DIR, CHROMA_DIR
from rag.loader import load_documents
from rag.chunker import chunk_documents
from rag.vectorstore import build_vectorstore


def main() -> None:
    print("=" * 50)
    print("  Ask Andela — Ingestion Pipeline")
    print("=" * 50)

    print(f"\n[1/3] Loading documents from {DATA_DIR} ...")
    documents = load_documents(DATA_DIR)
    print(f"      {len(documents)} documents loaded")
    for doc in documents:
        print(f"        · {doc['metadata']['source']}  ({len(doc['content']):,} chars)")

    print(f"\n[2/3] Chunking ...")
    chunks = chunk_documents(documents)
    print(f"      {len(chunks)} chunks created")

    print(f"\n[3/3] Building vector store at {CHROMA_DIR} ...")
    build_vectorstore(chunks, CHROMA_DIR)

    print("\n✓  Done. Run `python scripts/app.py` to start the UI.")


if __name__ == "__main__":
    main()
