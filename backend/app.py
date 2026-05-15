"""
app.py — FastAPI application for the Legal AI System.

Endpoints:
  GET  /health                  — System health check
  POST /ingest                  — Upload + process a document
  GET  /documents               — List all ingested documents
  GET  /documents/{doc_id}      — Get document metadata
  DELETE /documents/{doc_id}    — Delete a document
  POST /draft                   — Generate a grounded draft
  GET  /drafts                  — List all drafts
  GET  /drafts/{draft_id}       — Get a single draft
  POST /edit                    — Submit operator edit (triggers learning)
  GET  /patterns                — List learned edit patterns
  GET  /edits                   — List recent operator edits
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend import db, generator
from backend.config import UPLOAD_DIR, validate
from backend.db import (
    delete_document,
    get_document,
    get_draft,
    get_patterns,
    get_recent_edits,
    init_db,
    list_documents,
    list_drafts,
    save_draft,
)
from backend.edit_learner import capture_and_learn
from backend.embeddings import delete_doc_chunks, health_check
from backend.ingest import ingest_file
from backend.logger import get_logger
from backend.schemas import (
    DraftRequest,
    DraftResponse,
    EditCapture,
    HealthResponse,
    IngestResponse,
)

log = get_logger(__name__)

# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Legal AI System — Pearson Specter Litt",
    description=(
        "Ingest messy legal documents, retrieve grounded evidence, "
        "generate structured drafts, and learn from operator edits."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed upload types
_ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


@app.on_event("startup")
def _startup() -> None:
    """Validate config and initialise the database on every startup."""
    try:
        validate()
    except ValueError as exc:
        log.critical("Startup config error: %s", exc)
        raise
    init_db()
    log.info("Legal AI System is running.")


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Returns the operational status of all system components."""
    chroma_ok = health_check()
    try:
        list_documents()
        db_ok = True
    except Exception:
        db_ok = False

    status = "ok" if (chroma_ok and db_ok) else "degraded"
    return HealthResponse(status=status, version="1.0.0", chroma_ok=chroma_ok, db_ok=db_ok)

# ── Config ────────────────────────────────────────────────────────────────────

@app.get("/config")
async def get_config() -> dict:
    """Return current system configuration (safe, non-secret values)."""
    from backend.config import (
        GEMINI_MODEL, EMBEDDING_MODEL, CHUNK_SIZE,
        CHUNK_OVERLAP, RETRIEVAL_TOP_K, MAX_OUTPUT_TOKENS,
        API_HOST, API_PORT,          # ← add these two
    )
    return {
        "llm_model": GEMINI_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "api_base": f"http://{API_HOST}:{API_PORT}",
    }


# ── Document Ingest ───────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload a PDF or image file and run the full ingestion pipeline.

    The file is saved to disk, then OCR → parse → chunk → embed → store.
    """
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {', '.join(sorted(_ALLOWED_SUFFIXES))}"
            ),
        )

    doc_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{doc_id}{suffix}"

    # Save the upload to disk
    try:
        with open(save_path, "wb") as fh:
            shutil.copyfileobj(file.file, fh)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File save failed: {exc}")

    # Run the pipeline
    try:
        result = ingest_file(save_path, doc_id=doc_id)
    except RuntimeError as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc))

    return result


# ── Documents ─────────────────────────────────────────────────────────────────

@app.get("/documents", tags=["Documents"])
def get_documents() -> List[dict]:
    """List all ingested documents (summary view)."""
    return list_documents()


@app.get("/documents/{doc_id}", tags=["Documents"])
def get_document_detail(doc_id: str) -> dict:
    """Return full metadata for one document."""
    doc = get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return doc


@app.delete("/documents/{doc_id}", tags=["Documents"])
def remove_document(doc_id: str) -> dict:
    """Delete a document from SQLite and remove its chunks from ChromaDB."""
    if get_document(doc_id) is None:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    delete_document(doc_id)
    n_deleted = delete_doc_chunks(doc_id)
    return {
        "message": f"Document '{doc_id}' deleted.",
        "chunks_removed": n_deleted,
    }


# ── Draft Generation ──────────────────────────────────────────────────────────

@app.post("/draft", response_model=DraftResponse, tags=["Drafts"])
def create_draft(request: DraftRequest) -> DraftResponse:
    """
    Generate a grounded legal draft for one or more documents.

    The draft is anchored to retrieved evidence — no hallucinated conclusions.
    """
    for doc_id in request.doc_ids:
        if get_document(doc_id) is None:
            raise HTTPException(
                status_code=404,
                detail=f"Document '{doc_id}' not found. Please ingest it first.",
            )

    try:
        draft = generator.generate_draft(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    save_draft(draft)
    return draft


@app.get("/drafts", tags=["Drafts"])
def get_drafts(doc_id: Optional[str] = None) -> List[dict]:
    """List all drafts, optionally filtered by document ID."""
    return list_drafts(doc_id=doc_id)


@app.get("/drafts/{draft_id}", tags=["Drafts"])
def get_draft_detail(draft_id: str) -> dict:
    """Return the full content of one draft."""
    draft = get_draft(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail=f"Draft '{draft_id}' not found.")
    return draft


# ── Edit Learning ─────────────────────────────────────────────────────────────

@app.post("/edit", tags=["Edit Learning"])
def submit_edit(capture: EditCapture) -> dict:
    """
    Submit an operator's edited version of a draft.

    The system diffs original vs edited, extracts a reusable instruction,
    and stores it to improve all future drafts of the same type.
    """
    draft_record = get_draft(capture.draft_id)
    if draft_record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Draft '{capture.draft_id}' not found.",
        )

    draft_type: str = draft_record.get("draft_type", "case_fact_summary")

    try:
        pattern = capture_and_learn(capture, draft_type=draft_type)
    except Exception as exc:
        log.error("Edit learning error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Edit learning failed: {exc}")

    return {
        "message": "Edit captured and pattern learned successfully.",
        "pattern_id": pattern.pattern_id,
        "description": pattern.description,
        "instruction": pattern.instruction,
        "frequency": pattern.frequency,
    }


@app.get("/patterns", tags=["Edit Learning"])
def list_patterns(draft_type: Optional[str] = None) -> List[dict]:
    """List all learned editing patterns, optionally filtered by draft type."""
    return get_patterns(draft_type=draft_type)


@app.get("/edits", tags=["Edit Learning"])
def list_edits(draft_type: Optional[str] = None, limit: int = 20) -> List[dict]:
    """List recent operator edits."""
    return get_recent_edits(draft_type=draft_type, limit=limit)


# ── Dev entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from backend.config import API_HOST, API_PORT, DEBUG

    uvicorn.run(
        "backend.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=DEBUG,
        log_level="info",
    )