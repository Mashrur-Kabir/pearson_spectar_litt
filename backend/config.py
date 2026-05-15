"""
config.py — Centralized configuration for the Legal AI System.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# ── LLM ───────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR: Path = BASE_DIR / "data"
UPLOAD_DIR: Path = DATA_DIR / "uploads"
CHROMA_DIR: Path = DATA_DIR / "chroma_db"
SQLITE_PATH: Path = DATA_DIR / "legal_ai.db"

for _d in (UPLOAD_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "legal_docs")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "20"))

# ── Generation ────────────────────────────────────────────────────────────────
MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.2"))

# ── OCR ───────────────────────────────────────────────────────────────────────
TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "tesseract")
OCR_DPI: int = int(os.getenv("OCR_DPI", "300"))

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"


def validate() -> None:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in .env")