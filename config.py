from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# -----------------------------
# Paths
# -----------------------------
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
VECTOR_DB_DIR = Path("db")
CHUNKS_FILE = PROCESSED_DIR / "chunks.pkl"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Embeddings model
# -----------------------------
EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

# -----------------------------
# OpenRouter LLM config
# -----------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set in your environment variables.")

OPENROUTER_MODEL = "openai/gpt-4o-mini"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
MAX_NEW_TOKENS = 300