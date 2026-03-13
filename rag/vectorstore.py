"""
Embedding model + ChromaDB vector store management.

The SentenceTransformer model is a module-level singleton - loaded once on
first use and reused across all calls, so startup cost is paid only once.
"""
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, COLLECTION_NAME, CHROMA_DIR

# ── Embedding model (singleton) ───────────────────────────────────────────────
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[vectorstore] Loading embedding model: {EMBEDDING_MODEL} ...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        print("[vectorstore] Embedding model ready.")
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed a list of strings; returns L2-normalised vectors (cosine space)."""
    return (
        _get_model()
        .encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        .tolist()
    )


# ── ChromaDB helpers ──────────────────────────────────────────────────────────

def build_vectorstore(
    chunks: list[dict],
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 100,
) -> chromadb.Collection:
    """
    Embed all chunks and write them into a new persistent ChromaDB collection.

    Drops and recreates the collection if it already exists, so this function
    always produces a fresh store from the current data.
    Call `load_vectorstore()` on subsequent runs to skip re-embedding.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"[vectorstore] Dropped existing collection '{collection_name}'")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"[vectorstore] Embedding {len(chunks)} chunks ...")
    t0 = time.time()

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            documents=[c["content"]  for c in batch],
            embeddings=embed_texts([c["content"] for c in batch]),
            metadatas=[c["metadata"] for c in batch],
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
        )

    elapsed = time.time() - t0
    print(f"[vectorstore] Done - {collection.count()} vectors in {elapsed:.1f}s")
    return collection


def load_vectorstore(
    persist_dir: Path = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
) -> chromadb.Collection:
    """Load an existing ChromaDB collection without re-embedding."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)
    print(f"[vectorstore] Loaded '{collection_name}': {collection.count()} vectors")
    return collection
