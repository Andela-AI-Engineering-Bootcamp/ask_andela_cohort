"""
Central configuration — all constants, paths, and env vars live here.
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
# Free-tier options on OpenRouter (no billing required):
#   "meta-llama/llama-3.1-8b-instruct"
#   "qwen/qwen-2.5-7b-instruct"
#   "mistralai/mistral-7b-instruct"
# Better quality (paid, still cheap):
#   "openai/gpt-4o-mini"   "anthropic/claude-3-haiku"
LLM_MODEL       = "meta-llama/llama-3.1-8b-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Vector store ──────────────────────────────────────────────────────────────
COLLECTION_NAME = "ask_andela"

# ── Chunking (per PRD spec) ───────────────────────────────────────────────────
CHUNK_SIZE    = 400   # tokens
CHUNK_OVERLAP = 50    # tokens

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 5
