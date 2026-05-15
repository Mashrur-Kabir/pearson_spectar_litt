"""
schemas.py — Pydantic models (request / response shapes) for the Legal AI System.

These are used by:
  - FastAPI for request validation and OpenAPI docs
  - Internal modules to keep data structured and typed
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Document ──────────────────────────────────────────────────────────────────

class DocumentMeta(BaseModel):
    """Metadata stored for every ingested file."""
    doc_id: str
    filename: str
    file_path: str
    page_count: int
    word_count: int
    ingested_at: datetime
    status: str = "processed"   # processed | failed | pending


class ExtractedText(BaseModel):
    """Raw text output from OCR / text extraction."""
    doc_id: str
    page_number: int
    raw_text: str
    confidence: Optional[float] = None   # OCR confidence 0-1 if available
    extraction_method: str = "unknown"   # pdfplumber | gemini_vision | tesseract | unstructured


class ParsedDocument(BaseModel):
    """Structured legal fields extracted by the parser."""
    doc_id: str
    filename: str
    document_type: str = "unknown"       # contract | notice | case_file | memo | unknown
    parties: List[str] = Field(default_factory=list)
    dates: List[str] = Field(default_factory=list)
    case_numbers: List[str] = Field(default_factory=list)
    key_clauses: List[str] = Field(default_factory=list)
    jurisdiction: Optional[str] = None
    summary_hint: Optional[str] = None  # First 500 chars used as generation hint
    full_text: str = ""


# ── Chunk / Retrieval ─────────────────────────────────────────────────────────

class Chunk(BaseModel):
    """A text chunk ready for embedding."""
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever, with its similarity score."""
    chunk_id: str
    doc_id: str
    filename: str
    chunk_index: int
    text: str
    score: float                # higher = more relevant (cosine similarity)
    page_number: Optional[int] = None


class RetrievalResult(BaseModel):
    chunks: List[RetrievedChunk]
    query: str


# ── Draft Generation ──────────────────────────────────────────────────────────

class DraftRequest(BaseModel):
    """Sent by the frontend / operator to request a draft."""
    doc_ids: List[str] = Field(..., min_length=1)
    query: str = Field(
        default="Generate a comprehensive legal case fact summary.",
        description="The drafting task / instruction.",
    )
    draft_type: str = Field(
        default="case_fact_summary",
        description=(
            "One of: case_fact_summary | title_review | notice_summary "
            "| document_checklist | internal_memo"
        ),
    )


class DraftResponse(BaseModel):
    """The generated draft returned to the operator."""
    draft_id: str
    doc_ids: List[str]
    draft_type: str
    content: str
    evidence: List[RetrievedChunk]   # grounding evidence shown to operator
    generated_at: datetime
    model_used: str


# ── Edit Learner ──────────────────────────────────────────────────────────────

class EditCapture(BaseModel):
    """Operator submits their edited version of a draft."""
    draft_id: str
    original_content: str
    edited_content: str
    operator_notes: Optional[str] = None


class EditPattern(BaseModel):
    """A reusable pattern extracted from one or more operator edits."""
    pattern_id: str
    description: str             # Human-readable: "Operator prefers bullet lists for clauses"
    instruction: str             # Injected into future prompts: "Format clauses as bullet points"
    draft_type: str
    frequency: int = 1           # How many edits contributed to this pattern
    created_at: datetime
    last_seen_at: datetime


class EditPatternList(BaseModel):
    patterns: List[EditPattern]


# ── API helpers ───────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    word_count: int
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    chroma_ok: bool
    db_ok: bool


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None