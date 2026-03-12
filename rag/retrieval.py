"""
Retrieval — embeds a query and fetches the top-K most similar chunks
from the ChromaDB collection.
"""
import chromadb

from .config import TOP_K
from .vectorstore import embed_texts


def retrieve(
    query: str,
    collection: chromadb.Collection,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Embed the query and return the top-K most relevant chunks.

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

    return [
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
