"""
Token-based text chunker using tiktoken.

Splits documents into overlapping chunks of CHUNK_SIZE tokens with CHUNK_OVERLAP
token overlap, as specified in the PRD.  Metadata is propagated to every chunk.
"""
import tiktoken

from .config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """Split a string into overlapping token-based chunks."""
    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += chunk_size - chunk_overlap

    return chunks


def chunk_documents(
    documents: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Chunk all documents and propagate their metadata to every chunk.

    Each chunk dict:
        {
            "content":  str,
            "metadata": {
                "source", "category",   ← inherited from parent doc
                "chunk_index",          ← 0-based position within the doc
                "total_chunks"          ← total chunks produced from this doc
            }
        }
    """
    all_chunks: list[dict] = []
    for doc in documents:
        text_chunks = chunk_text(doc["content"], chunk_size, chunk_overlap)
        for i, chunk in enumerate(text_chunks):
            all_chunks.append({
                "content": chunk,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index":  i,
                    "total_chunks": len(text_chunks),
                },
            })
    return all_chunks
