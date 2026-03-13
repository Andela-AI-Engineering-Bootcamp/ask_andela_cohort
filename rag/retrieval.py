"""
Retrieval - embeds a query and fetches the top-K most similar chunks
from the ChromaDB collection.

Retrieval always uses the raw query so that prior conversation topics
cannot bleed into the embedding and pull results toward the wrong source.
Conversation history is handled exclusively at the LLM generation step,
where the model can reason about follow-ups without contaminating retrieval.
"""
import chromadb

from .config import TOP_K, RELEVANCE_THRESHOLD
from .vectorstore import embed_texts


def retrieve(
    query: str,
    collection: chromadb.Collection,
    top_k: int = TOP_K,
    history: list[dict] | None = None,
    threshold: float = RELEVANCE_THRESHOLD,
) -> list[dict]:
    """
    Embed the raw query and return the top-K most relevant chunks,
    filtered by relevance threshold.

    `history` is accepted for interface compatibility but intentionally
    not used in the embedding step — injecting prior turns into the query
    vector causes topic bleed when the user switches subjects mid-chat
    (e.g. prior "2-day group project" turns contaminating a "final project"
    retrieval).  Conversation context is handled by the LLM instead.

    Always returns at least 3 chunks to avoid an empty context window,
    even if all scores are below threshold.

    Each result dict:
        {
            "content":         str,    # raw chunk text
            "source":          str,    # source filename — used as citation
            "category":        str,    # discourse_channel | course_resource | …
            "chunk_index":     int,
            "relevance_score": float   # 0–1 cosine similarity, higher = better
        }
    """
    results = collection.query(
        query_embeddings=embed_texts([query]),
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = [
        {
            "content":         doc,
            "source":          meta.get("source", "unknown"),
            "category":        meta.get("category", "unknown"),
            "chunk_index":     meta.get("chunk_index", 0),
            "relevance_score": round(1.0 - dist, 4),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

    # Drop low-quality chunks; always keep at least 3 to avoid empty context
    above_threshold = [c for c in chunks if c["relevance_score"] >= threshold]
    return above_threshold if len(above_threshold) >= 3 else chunks[:3]
