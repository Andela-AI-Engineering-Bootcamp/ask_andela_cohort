"""
Full RAG pipeline — the single public function the UI and eval scripts call.

    ask_andela(query, collection) → {"answer", "sources", "context_chunks"}
"""
import chromadb

from .config import TOP_K, LLM_MODEL
from .retrieval import retrieve
from .llm import generate_answer


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
