"""
Central configuration - all constants, paths, and env vars live here.
Every other module imports from this file; nothing else calls load_dotenv().
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
# This file lives at:  <project_root>/rag/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CHROMA_DIR   = PROJECT_ROOT / "chroma_db"

load_dotenv(PROJECT_ROOT / ".env")

# ── API keys ──────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")

if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "OPENROUTER_API_KEY is not set. "
        f"Add it to {PROJECT_ROOT / '.env'} or export it as an environment variable."
    )

# ── Models ────────────────────────────────────────────────────────────────────
# Experiment with llama-3.1-8b-instruct and qwen-2.5-72b-instruct produced slow and 
# sub-par results, hence the choice of gemini-2.0-flash-lite.
# Upgrading to the paid models will increase the cost of the RAG pipeline.
LLM_MODEL = "google/gemini-2.5-flash-lite"

# Embedding - bge-small-en-v1.5 is the same size as all-MiniLM-L6-v2 (384-dim,
# same speed & memory) but trained for asymmetric retrieval (question -> passage)
# rather than symmetric sentence similarity, so it scores topically-related but
# differently-phrased content much higher.
# NOTE: Changing this requires re-running ingest.py to rebuild the ChromaDB vectors.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# ── Vector store ──────────────────────────────────────────────────────────────
COLLECTION_NAME = "ask_andela"

# ── Chunking (per PRD spec) ───────────────────────────────────────────────────
CHUNK_SIZE    = 400   # tokens
CHUNK_OVERLAP = 50    # tokens

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K               = 8     # fetch more chunks; broad questions span multiple sources
RELEVANCE_THRESHOLD = 0.25  # drop chunks below this cosine similarity to reduce noise
# BGE models score asymmetric retrieval (question -> passage) accurately enough
# that 0.25 is a reliable signal.  With the old all-MiniLM-L6-v2 model this had
# to be lowered to 0.15 because STS-trained models under-score valid passages.
MAX_HISTORY_TURNS   = 3     # conversation turns (user+assistant pairs) sent to LLM
