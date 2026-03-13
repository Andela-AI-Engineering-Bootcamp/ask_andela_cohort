"""
Full RAG pipeline.

Public API:
    ask_andela(query, collection)    -> {"answer", "sources", "context_chunks"}  (blocking)
    stream_andela(query, collection) -> Iterator[dict]                            (streaming)

stream_andela yields two event shapes in order:
    {"type": "sources", "sources": list[str], "context_chunks": list[dict]}
    {"type": "token",   "token":   str}

The sources event fires as soon as retrieval completes (before any tokens arrive),
so the UI can populate the sources panel without waiting for the full answer.
"""
from collections.abc import Iterator

import chromadb

from .config import TOP_K, LLM_MODEL
from .retrieval import retrieve
from .llm import generate_answer, stream_answer


def ask_andela(
    query: str,
    collection: chromadb.Collection,
    top_k: int = TOP_K,
    model: str = LLM_MODEL,
    history: list[dict] | None = None,
) -> dict:
    """
    End-to-end RAG pipeline:
        1. Contextualize + embed the query (using prior conversation history)
        2. Retrieve top-K chunks from ChromaDB, filtered by relevance threshold
        3. Assemble prompt + history and call LLM via OpenRouter
        4. Return answer + deduplicated source citations

    Args:
        history: Prior conversation turns as list of {"role", "content"} dicts.
                 Pass the turns *before* the current query so the model can
                 resolve follow-up questions correctly.

    Returns:
        {
            "answer":         str,          # LLM-generated answer
            "sources":        list[str],    # source filenames, best-match first
            "context_chunks": list[dict],   # raw chunks (useful for debugging)
        }
    """
    context_chunks = retrieve(query, collection, top_k, history=history)
    answer = generate_answer(query, context_chunks, model, history=history)

    # Deduplicate sources, preserving relevance order
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in sorted(context_chunks, key=lambda c: c["relevance_score"], reverse=True):
        if chunk["source"] not in seen:
            sources.append(chunk["source"])
            seen.add(chunk["source"])

    return {
        "answer":         answer,
        "sources":        sources,
        "context_chunks": context_chunks,
    }


def stream_andela(
    query: str,
    collection: chromadb.Collection,
    top_k: int = TOP_K,
    model: str = LLM_MODEL,
    history: list[dict] | None = None,
) -> Iterator[dict]:
    """
    Streaming RAG pipeline.

    Yields events in two phases:
        1. A single "sources" event immediately after retrieval completes,
           so the UI can show citations before the first token arrives.
        2. One "token" event per streamed text delta from the LLM.

    Args:
        history: Prior conversation turns as list of {"role", "content"} dicts.

    Yields:
        {"type": "sources", "sources": list[str], "context_chunks": list[dict]}
        {"type": "token",   "token":   str}
    """
    context_chunks = retrieve(query, collection, top_k, history=history)

    seen: set[str] = set()
    sources: list[str] = []
    for chunk in sorted(context_chunks, key=lambda c: c["relevance_score"], reverse=True):
        if chunk["source"] not in seen:
            sources.append(chunk["source"])
            seen.add(chunk["source"])

    yield {"type": "sources", "sources": sources, "context_chunks": context_chunks}

    for token in stream_answer(query, context_chunks, model, history=history):
        yield {"type": "token", "token": token}
