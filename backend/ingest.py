"""
ingest.py — Full document ingestion pipeline orchestrator.

Stages (in order):
  1. OCR / text extraction  (ocr.py)
  2. Structured field parsing  (parser.py)
  3. Text chunking  (chunker.py)
  4. Embedding + ChromaDB indexing  (embeddings.py)
  5. Metadata persistence  (db.py)

Each stage logs progress and raises RuntimeError on failure so the
API layer can return an informative error to the caller.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend import ocr, parser
from backend.chunker import chunk_document
from backend.config import UPLOAD_DIR
from backend.db import init_db, save_document
from backend.embeddings import add_chunks
from backend.logger import get_logger
from backend.schemas import DocumentMeta, IngestResponse

log = get_logger(__name__)


def ingest_file(
    file_path: Path,
    doc_id: Optional[str] = None,
) -> IngestResponse:
    """
    Run the complete ingestion pipeline on a single uploaded file.

    Args:
        file_path: Absolute path to the saved file (PDF or image).
        doc_id:    Optional ID override. Auto-generated (UUID4) if omitted.

    Returns:
        IngestResponse with status, doc_id, and processing summary.

    Raises:
        RuntimeError: If any pipeline stage fails.
    """
    if doc_id is None:
        doc_id = str(uuid.uuid4())

    log.info("─── Ingestion START | file=%s | doc_id=%s ───", file_path.name, doc_id)

    # ── Stage 1: OCR / text extraction ───────────────────────────────────────
    log.info("[1/5] Extracting text from %s", file_path.name)
    try:
        extracted_pages = ocr.extract(file_path, doc_id)
    except Exception as exc:
        log.error("OCR failed: %s", exc)
        raise RuntimeError(f"Text extraction failed: {exc}") from exc

    if not extracted_pages:
        raise RuntimeError(f"No text could be extracted from '{file_path.name}'.")

    total_chars = sum(len(p.raw_text) for p in extracted_pages)
    log.info(
        "[1/5] Done — %d pages, %d total characters.",
        len(extracted_pages), total_chars,
    )

    # ── Stage 2: Parse structured fields ─────────────────────────────────────
    log.info("[2/5] Parsing structured legal fields.")
    try:
        parsed = parser.parse(extracted_pages, filename=file_path.name)
    except Exception as exc:
        log.error("Parsing failed: %s", exc)
        raise RuntimeError(f"Parsing failed: {exc}") from exc

    word_count = len(parsed.full_text.split())
    log.info(
        "[2/5] Done — type=%s | words=%d | parties=%d | clauses=%d.",
        parsed.document_type, word_count,
        len(parsed.parties), len(parsed.key_clauses),
    )

    # ── Stage 3: Chunk ────────────────────────────────────────────────────────
    log.info("[3/5] Chunking document text.")
    try:
        chunks = chunk_document(parsed)
    except Exception as exc:
        log.error("Chunking failed: %s", exc)
        raise RuntimeError(f"Chunking failed: {exc}") from exc

    log.info("[3/5] Done — %d chunks created.", len(chunks))

    # ── Stage 4: Embed + index ────────────────────────────────────────────────
    log.info("[4/5] Embedding and indexing chunks into ChromaDB.")
    try:
        n_added = add_chunks(chunks)
    except Exception as exc:
        log.error("Embedding failed: %s", exc)
        raise RuntimeError(f"Embedding/indexing failed: {exc}") from exc

    log.info("[4/5] Done — %d chunks indexed.", n_added)

    # ── Stage 5: Persist metadata to SQLite ──────────────────────────────────
    log.info("[5/5] Saving document metadata to SQLite.")
    meta = DocumentMeta(
        doc_id=doc_id,
        filename=file_path.name,
        file_path=str(file_path),
        page_count=len(extracted_pages),
        word_count=word_count,
        ingested_at=datetime.now(timezone.utc),
        status="processed",
    )
    try:
        init_db()          # idempotent: creates tables if they don't exist
        save_document(meta, parsed)
    except Exception as exc:
        log.error("DB save failed: %s", exc)
        raise RuntimeError(f"Database save failed: {exc}") from exc

    log.info(
        "─── Ingestion COMPLETE | doc_id=%s | pages=%d | chunks=%d ───",
        doc_id, len(extracted_pages), len(chunks),
    )

    return IngestResponse(
        doc_id=doc_id,
        filename=file_path.name,
        page_count=len(extracted_pages),
        word_count=word_count,
        status="processed",
        message=(
            f"Successfully ingested '{file_path.name}': "
            f"{len(extracted_pages)} pages, {len(chunks)} chunks indexed."
        ),
    )