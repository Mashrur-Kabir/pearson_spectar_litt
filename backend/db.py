"""
db.py — Adaptive persistence layer for the Legal AI System.

When SUPABASE_URL + SUPABASE_KEY are set in .env: uses Supabase (PostgreSQL) [Production].
When they are NOT set: falls back to local SQLite [Local Dev / Testing].

Tables:
  documents     — every ingested file's metadata + full text
  drafts        — every generated draft
  edits         — operator-submitted edits
  edit_patterns — learned improvement patterns
  users         — user authentication (Supabase only)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from backend.config import SUPABASE_URL, SUPABASE_KEY, SQLITE_PATH
from backend.logger import get_logger
from backend.schemas import (
    DocumentMeta,
    DraftResponse,
    EditCapture,
    EditPattern,
    ParsedDocument,
)

log = get_logger(__name__)

# ── Client Selection ──────────────────────────────────────────────────────────

USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY)

if USE_SUPABASE:
    from supabase import create_client, Client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    log.info("Using Supabase (PostgreSQL) as the database backend.")
else:
    supabase = None
    log.warning("SUPABASE_URL/KEY not set — falling back to local SQLite at %s", SQLITE_PATH)

# ── SQLite internals (used only in local dev mode) ────────────────────────────

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(SQLITE_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    file_path    TEXT NOT NULL,
    page_count   INTEGER DEFAULT 0,
    word_count   INTEGER DEFAULT 0,
    document_type TEXT DEFAULT 'unknown',
    parties      TEXT DEFAULT '[]',
    dates        TEXT DEFAULT '[]',
    case_numbers TEXT DEFAULT '[]',
    key_clauses  TEXT DEFAULT '[]',
    jurisdiction TEXT,
    full_text    TEXT DEFAULT '',
    ingested_at  TEXT NOT NULL,
    status       TEXT DEFAULT 'processed'
);

CREATE TABLE IF NOT EXISTS drafts (
    draft_id      TEXT PRIMARY KEY,
    doc_ids       TEXT NOT NULL,
    draft_type    TEXT NOT NULL,
    content       TEXT NOT NULL,
    evidence_json TEXT DEFAULT '[]',
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
    """Initialise local SQLite tables (no-op for Supabase)."""
    if not USE_SUPABASE:
        with get_db() as conn:
            conn.executescript(_SCHEMA_SQL)
        log.info("SQLite database initialised at %s", SQLITE_PATH)
    else:
        log.info("Supabase connection ready.")


# ── Documents ─────────────────────────────────────────────────────────────────

def save_document(meta: DocumentMeta, parsed: ParsedDocument) -> None:
    if USE_SUPABASE:
        data = {
            "doc_id": meta.doc_id, "filename": meta.filename,
            "file_path": meta.file_path, "page_count": meta.page_count,
            "word_count": meta.word_count, "document_type": parsed.document_type,
            "parties": parsed.parties, "dates": parsed.dates,
            "case_numbers": parsed.case_numbers, "key_clauses": parsed.key_clauses,
            "jurisdiction": parsed.jurisdiction, "full_text": parsed.full_text,
            "ingested_at": meta.ingested_at.isoformat(), "status": meta.status,
        }
        supabase.table("documents").upsert(data).execute()
    else:
        sql = """
        INSERT OR REPLACE INTO documents
            (doc_id, filename, file_path, page_count, word_count,
             document_type, parties, dates, case_numbers, key_clauses,
             jurisdiction, full_text, ingested_at, status)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        with get_db() as conn:
            conn.execute(sql, (
                meta.doc_id, meta.filename, meta.file_path,
                meta.page_count, meta.word_count, parsed.document_type,
                json.dumps(parsed.parties), json.dumps(parsed.dates),
                json.dumps(parsed.case_numbers), json.dumps(parsed.key_clauses),
                parsed.jurisdiction, parsed.full_text,
                meta.ingested_at.isoformat(), meta.status,
            ))
    log.debug("Saved document %s (%s)", meta.doc_id, meta.filename)


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    if USE_SUPABASE:
        res = supabase.table("documents").select("*").eq("doc_id", doc_id).execute()
        return res.data[0] if res.data else None
    else:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        for field in ("parties", "dates", "case_numbers", "key_clauses"):
            d[field] = json.loads(d[field])
        return d


def list_documents() -> List[Dict[str, Any]]:
    if USE_SUPABASE:
        res = supabase.table("documents").select(
            "doc_id, filename, page_count, word_count, document_type, ingested_at, status"
        ).order("ingested_at", desc=True).execute()
        return res.data
    else:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT doc_id, filename, page_count, word_count, "
                "document_type, ingested_at, status FROM documents ORDER BY ingested_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]


def delete_document(doc_id: str) -> bool:
    if USE_SUPABASE:
        res = supabase.table("documents").delete().eq("doc_id", doc_id).execute()
        return len(res.data) > 0
    else:
        with get_db() as conn:
            cur = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        return cur.rowcount > 0


# ── Drafts ────────────────────────────────────────────────────────────────────

def save_draft(draft: DraftResponse) -> None:
    if USE_SUPABASE:
        data = {
            "draft_id": draft.draft_id, "doc_ids": draft.doc_ids,
            "draft_type": draft.draft_type, "content": draft.content,
            "evidence_json": [e.model_dump() for e in draft.evidence],
            "model_used": draft.model_used, "generated_at": draft.generated_at.isoformat(),
        }
        supabase.table("drafts").upsert(data).execute()
    else:
        sql = """
        INSERT OR REPLACE INTO drafts
            (draft_id, doc_ids, draft_type, content, evidence_json, model_used, generated_at)
        VALUES (?,?,?,?,?,?,?)
        """
        with get_db() as conn:
            conn.execute(sql, (
                draft.draft_id, json.dumps(draft.doc_ids), draft.draft_type,
                draft.content, json.dumps([e.model_dump() for e in draft.evidence]),
                draft.model_used, draft.generated_at.isoformat(),
            ))
    log.debug("Saved draft %s", draft.draft_id)


def get_draft(draft_id: str) -> Optional[Dict[str, Any]]:
    if USE_SUPABASE:
        res = supabase.table("drafts").select("*").eq("draft_id", draft_id).execute()
        return res.data[0] if res.data else None
    else:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM drafts WHERE draft_id = ?", (draft_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["doc_ids"] = json.loads(d["doc_ids"])
        d["evidence_json"] = json.loads(d["evidence_json"])
        return d


def list_drafts(doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if USE_SUPABASE:
        query = supabase.table("drafts").select(
            "draft_id, doc_ids, draft_type, generated_at"
        ).order("generated_at", desc=True)
        if doc_id:
            query = query.contains("doc_ids", f'["{doc_id}"]')
        return query.execute().data
    else:
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
    now = datetime.now(timezone.utc).isoformat()
    if USE_SUPABASE:
        supabase.table("edits").insert({
            "edit_id": edit_id, "draft_id": capture.draft_id,
            "original_content": capture.original_content,
            "edited_content": capture.edited_content,
            "operator_notes": capture.operator_notes, "captured_at": now,
        }).execute()
    else:
        sql = """
        INSERT INTO edits
            (edit_id, draft_id, original_content, edited_content, operator_notes, captured_at)
        VALUES (?,?,?,?,?,?)
        """
        with get_db() as conn:
            conn.execute(sql, (
                edit_id, capture.draft_id, capture.original_content,
                capture.edited_content, capture.operator_notes, now,
            ))
    log.debug("Saved edit %s for draft %s", edit_id, capture.draft_id)


def get_recent_edits(draft_type: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    if USE_SUPABASE:
        if draft_type:
            res = supabase.table("edits").select(
                "*, drafts!inner(draft_type)"
            ).eq("drafts.draft_type", draft_type).order("captured_at", desc=True).limit(limit).execute()
        else:
            res = supabase.table("edits").select("*").order("captured_at", desc=True).limit(limit).execute()
        return res.data
    else:
        with get_db() as conn:
            if draft_type:
                rows = conn.execute(
                    "SELECT e.* FROM edits e JOIN drafts d ON e.draft_id = d.draft_id "
                    "WHERE d.draft_type = ? ORDER BY e.captured_at DESC LIMIT ?",
                    (draft_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM edits ORDER BY captured_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(r) for r in rows]


# ── Edit Patterns ─────────────────────────────────────────────────────────────

def save_pattern(pattern: EditPattern) -> None:
    if USE_SUPABASE:
        supabase.table("edit_patterns").upsert({
            "pattern_id": pattern.pattern_id, "description": pattern.description,
            "instruction": pattern.instruction, "draft_type": pattern.draft_type,
            "frequency": pattern.frequency, "created_at": pattern.created_at.isoformat(),
            "last_seen_at": pattern.last_seen_at.isoformat(),
        }).execute()
    else:
        sql = """
        INSERT OR REPLACE INTO edit_patterns
            (pattern_id, description, instruction, draft_type, frequency, created_at, last_seen_at)
        VALUES (?,?,?,?,?,?,?)
        """
        with get_db() as conn:
            conn.execute(sql, (
                pattern.pattern_id, pattern.description, pattern.instruction,
                pattern.draft_type, pattern.frequency,
                pattern.created_at.isoformat(), pattern.last_seen_at.isoformat(),
            ))
    log.debug("Saved pattern %s", pattern.pattern_id)


def get_patterns(draft_type: Optional[str] = None) -> List[Dict[str, Any]]:
    if USE_SUPABASE:
        query = supabase.table("edit_patterns").select("*").order("frequency", desc=True)
        if draft_type:
            query = query.eq("draft_type", draft_type)
        return query.execute().data
    else:
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
    if USE_SUPABASE:
        res = supabase.table("edit_patterns").select("frequency").eq("pattern_id", pattern_id).execute()
        if res.data:
            supabase.table("edit_patterns").update({
                "frequency": res.data[0]["frequency"] + 1,
                "last_seen_at": datetime.now(timezone.utc).isoformat()
            }).eq("pattern_id", pattern_id).execute()
    else:
        with get_db() as conn:
            conn.execute(
                "UPDATE edit_patterns SET frequency = frequency + 1, last_seen_at = ? "
                "WHERE pattern_id = ?",
                (datetime.now(timezone.utc).isoformat(), pattern_id),
            )


# ── Users (Auth) ──────────────────────────────────────────────────────────────

def get_all_users() -> List[Dict[str, Any]]:
    """Return user list from Supabase, or a local test admin if in dev mode."""
    if USE_SUPABASE:
        res = supabase.table("users").select("*").execute()
        return res.data
    else:
        # Local dev: return a mock admin so the login screen can be tested.
        # Password is: admin123
        import bcrypt
        hashed = bcrypt.hashpw(b"admin123", bcrypt.gensalt()).decode()
        log.warning("Using mock local admin user for dev testing (username: admin, password: admin123)")
        return [{
            "username": "admin",
            "email": "admin@pearsonspecterlitt.com",
            "name": "Harvey Specter",
            "password_hash": hashed,
        }]