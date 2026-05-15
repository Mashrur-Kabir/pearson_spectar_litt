"""
db.py — SQLite persistence layer for the Legal AI System.

Tables:
  documents     — every ingested file's metadata + full text
  drafts        — every generated draft
  edits         — operator-submitted edits
  edit_patterns — learned improvement patterns

Uses the standard library `sqlite3` only (no ORM), so there are
zero extra dependencies beyond Python itself.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from backend.config import SQLITE_PATH
from backend.logger import get_logger
from backend.schemas import (
    DocumentMeta,
    DraftResponse,
    EditCapture,
    EditPattern,
    ParsedDocument,
)

log = get_logger(__name__)

# Thread-local connections so FastAPI workers don't share a single connection
_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(SQLITE_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")   # better concurrency
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a connection and commits on success."""
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ── Schema initialisation ─────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    file_path    TEXT NOT NULL,
    page_count   INTEGER DEFAULT 0,
    word_count   INTEGER DEFAULT 0,
    document_type TEXT DEFAULT 'unknown',
    parties      TEXT DEFAULT '[]',        -- JSON list
    dates        TEXT DEFAULT '[]',        -- JSON list
    case_numbers TEXT DEFAULT '[]',        -- JSON list
    key_clauses  TEXT DEFAULT '[]',        -- JSON list
    jurisdiction TEXT,
    full_text    TEXT DEFAULT '',
    ingested_at  TEXT NOT NULL,
    status       TEXT DEFAULT 'processed'
);

CREATE TABLE IF NOT EXISTS drafts (
    draft_id      TEXT PRIMARY KEY,
    doc_ids       TEXT NOT NULL,           -- JSON list
    draft_type    TEXT NOT NULL,
    content       TEXT NOT NULL,
    evidence_json TEXT DEFAULT '[]',       -- JSON list of RetrievedChunk
    model_used    TEXT NOT NULL,
    generated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS edits (
    edit_id          TEXT PRIMARY KEY,
    draft_id         TEXT NOT NULL,
    original_content TEXT NOT NULL,
    edited_content   TEXT NOT NULL,
    operator_notes   TEXT,
    captured_at      TEXT NOT NULL,
    FOREIGN KEY (draft_id) REFERENCES drafts(draft_id)
);

CREATE TABLE IF NOT EXISTS edit_patterns (
    pattern_id   TEXT PRIMARY KEY,
    description  TEXT NOT NULL,
    instruction  TEXT NOT NULL,
    draft_type   TEXT NOT NULL,
    frequency    INTEGER DEFAULT 1,
    created_at   TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);
"""


def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    with get_db() as conn:
        conn.executescript(_SCHEMA_SQL)
    log.info("Database initialised at %s", SQLITE_PATH)


# ── Documents ─────────────────────────────────────────────────────────────────

def save_document(meta: DocumentMeta, parsed: ParsedDocument) -> None:
    sql = """
    INSERT OR REPLACE INTO documents
        (doc_id, filename, file_path, page_count, word_count,
         document_type, parties, dates, case_numbers, key_clauses,
         jurisdiction, full_text, ingested_at, status)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    with get_db() as conn:
        conn.execute(sql, (
            meta.doc_id,
            meta.filename,
            meta.file_path,
            meta.page_count,
            meta.word_count,
            parsed.document_type,
            json.dumps(parsed.parties),
            json.dumps(parsed.dates),
            json.dumps(parsed.case_numbers),
            json.dumps(parsed.key_clauses),
            parsed.jurisdiction,
            parsed.full_text,
            meta.ingested_at.isoformat(),
            meta.status,
        ))
    log.debug("Saved document %s (%s)", meta.doc_id, meta.filename)


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    for field in ("parties", "dates", "case_numbers", "key_clauses"):
        d[field] = json.loads(d[field])
    return d


def list_documents() -> List[Dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT doc_id, filename, page_count, word_count, "
            "document_type, ingested_at, status FROM documents ORDER BY ingested_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(doc_id: str) -> bool:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
    return cur.rowcount > 0


# ── Drafts ────────────────────────────────────────────────────────────────────

def save_draft(draft: DraftResponse) -> None:
    sql = """
    INSERT OR REPLACE INTO drafts
        (draft_id, doc_ids, draft_type, content, evidence_json, model_used, generated_at)
    VALUES (?,?,?,?,?,?,?)
    """
    with get_db() as conn:
        conn.execute(sql, (
            draft.draft_id,
            json.dumps(draft.doc_ids),
            draft.draft_type,
            draft.content,
            json.dumps([e.model_dump() for e in draft.evidence]),
            draft.model_used,
            draft.generated_at.isoformat(),
        ))
    log.debug("Saved draft %s", draft.draft_id)


def get_draft(draft_id: str) -> Optional[Dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM drafts WHERE draft_id = ?", (draft_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["doc_ids"] = json.loads(d["doc_ids"])
    d["evidence_json"] = json.loads(d["evidence_json"])
    return d


def list_drafts(doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_db() as conn:
        if doc_id:
            rows = conn.execute(
                "SELECT draft_id, doc_ids, draft_type, generated_at "
                "FROM drafts WHERE doc_ids LIKE ? ORDER BY generated_at DESC",
                (f"%{doc_id}%",),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT draft_id, doc_ids, draft_type, generated_at "
                "FROM drafts ORDER BY generated_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]


# ── Edits ─────────────────────────────────────────────────────────────────────

def save_edit(edit_id: str, capture: EditCapture) -> None:
    sql = """
    INSERT INTO edits
        (edit_id, draft_id, original_content, edited_content, operator_notes, captured_at)
    VALUES (?,?,?,?,?,?)
    """
    with get_db() as conn:
        conn.execute(sql, (
            edit_id,
            capture.draft_id,
            capture.original_content,
            capture.edited_content,
            capture.operator_notes,
            datetime.now(timezone.utc).isoformat(),
        ))
    log.debug("Saved edit %s for draft %s", edit_id, capture.draft_id)


def get_recent_edits(draft_type: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    with get_db() as conn:
        if draft_type:
            rows = conn.execute(
                """
                SELECT e.* FROM edits e
                JOIN drafts d ON e.draft_id = d.draft_id
                WHERE d.draft_type = ?
                ORDER BY e.captured_at DESC LIMIT ?
                """,
                (draft_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM edits ORDER BY captured_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [dict(r) for r in rows]


# ── Edit Patterns ─────────────────────────────────────────────────────────────

def save_pattern(pattern: EditPattern) -> None:
    sql = """
    INSERT OR REPLACE INTO edit_patterns
        (pattern_id, description, instruction, draft_type, frequency, created_at, last_seen_at)
    VALUES (?,?,?,?,?,?,?)
    """
    with get_db() as conn:
        conn.execute(sql, (
            pattern.pattern_id,
            pattern.description,
            pattern.instruction,
            pattern.draft_type,
            pattern.frequency,
            pattern.created_at.isoformat(),
            pattern.last_seen_at.isoformat(),
        ))
    log.debug("Saved pattern %s", pattern.pattern_id)


def get_patterns(draft_type: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_db() as conn:
        if draft_type:
            rows = conn.execute(
                "SELECT * FROM edit_patterns WHERE draft_type = ? ORDER BY frequency DESC",
                (draft_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM edit_patterns ORDER BY frequency DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def increment_pattern_frequency(pattern_id: str) -> None:
    with get_db() as conn:
        conn.execute(
            "UPDATE edit_patterns SET frequency = frequency + 1, last_seen_at = ? "
            "WHERE pattern_id = ?",
            (datetime.now(timezone.utc).isoformat(), pattern_id),
        )