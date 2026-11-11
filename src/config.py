"""Configuration settings for the RAG with Redis Caching application."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Document processing settings
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))

# Vector database settings
CHROMA_PERSIST_DIR = str(BASE_DIR / "data" / "chroma_db")
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "rag_documents")
TOP_K_RETRIEVAL = int(os.getenv("RAG_TOP_K", "3"))

# Redis caching settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_URL = os.getenv("REDIS_URL", f"redis://{REDIS_HOST}:{REDIS_PORT}/0")
CACHE_TTL_SECONDS = int(os.getenv("RAG_CACHE_TTL", "3600"))  # Default: 1 hour
CACHE_NAMESPACE = "rag:cache"

# LLM provider settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower() or "gemini"
_raw_model = os.getenv("LLM_MODEL", "").strip() or "gemini-flash-lite-latest"
_MODEL_ALIASES = {
    "gemini-flash-lite-latest": "gemini-flash-lite-latest",
    "gemini-flash-lite": "gemini-flash-lite-latest",
    "gemini-flash-latest": "gemini-flash-latest",
    "gemini-flash": "gemini-flash-lite-latest",
    "gemini-1.5-flash": "gemini-flash-lite-latest",
    "gemini-2.5-flash": "gemini-flash-lite-latest",
    "gemini-3-flash": "gemini-3-flash-preview",
}
DEFAULT_MODEL = _MODEL_ALIASES.get(_raw_model.lower(), _raw_model)
PROMPT_VERSION = "v2.0"  # Bumped to v2.0 to ensure zero stale HTML-polluted cache entries
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
