"""
parser.py — Extracts structured legal fields from raw OCR/extracted text.

Uses a combination of:
  - regex patterns (fast, deterministic)
  - keyword heuristics (document type classification)

Output: ParsedDocument (see schemas.py)
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import List, Optional

from backend.logger import get_logger
from backend.schemas import ExtractedText, ParsedDocument

log = get_logger(__name__)


# ── Regex patterns ────────────────────────────────────────────────────────────

# Dates: various common legal date formats
_DATE_PATTERNS = [
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",                          # 01/15/2024
    r"\b(\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b",                        # 15 January 2024
    r"\b((?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4})\b",             # January 15, 2024
    r"\b(\d{4}-\d{2}-\d{2})\b",                                        # 2024-01-15
]

# Case / docket numbers
_CASE_PATTERNS = [
    r"\bCase\s+No\.?\s*([\w\-/]+)",
    r"\bDocket\s+No\.?\s*([\w\-/]+)",
    r"\bC\.A\.\s+No\.?\s*([\w\-/]+)",
    r"\bCV[-\s]?(\d[\w\-/]+)",
    r"\bCR[-\s]?(\d[\w\-/]+)",
    r"\bNo\.\s+(\d[\d\-/]+)",
]

# Party names — looks for "Plaintiff:", "Defendant:", "Between:", "Petitioner:", etc.
_PARTY_PATTERNS = [
    r"(?:Plaintiff|PLAINTIFF)\s*[:\-]\s*(.+?)(?:\n|;|,\s+and\b)",
    r"(?:Defendant|DEFENDANT)\s*[:\-]\s*(.+?)(?:\n|;|,\s+and\b)",
    r"(?:Petitioner|PETITIONER)\s*[:\-]\s*(.+?)(?:\n|;)",
    r"(?:Respondent|RESPONDENT)\s*[:\-]\s*(.+?)(?:\n|;)",
    r"(?:Claimant|CLAIMANT)\s*[:\-]\s*(.+?)(?:\n|;)",
    r"(?:v\.|vs\.?|versus)\s+(.+?)(?:\n|,|\()",
]

# Jurisdiction
_JURISDICTION_PATTERNS = [
    r"(?:State of|Court of|District of|Superior Court)\s+([A-Z][a-zA-Z\s]+?)(?:\n|,|\.|;)",
    r"IN THE (?:UNITED STATES )?(?:DISTRICT|SUPERIOR|CIRCUIT|APPELLATE|SUPREME) COURT"
    r"(?:\s+FOR\s+THE\s+)?([A-Z ]+?)(?:\n|,|\.|DIVISION)",
]

# Key clause triggers (section headings often used in legal docs)
_CLAUSE_KEYWORDS = [
    "WHEREAS", "NOW THEREFORE", "CONSIDERATION", "INDEMNIFICATION",
    "LIMITATION OF LIABILITY", "GOVERNING LAW", "JURISDICTION", "TERMINATION",
    "CONFIDENTIALITY", "NON-DISCLOSURE", "REPRESENTATIONS AND WARRANTIES",
    "DISPUTE RESOLUTION", "ARBITRATION", "FORCE MAJEURE", "ASSIGNMENT",
    "ENTIRE AGREEMENT", "AMENDMENT", "SEVERABILITY", "NOTICE",
]

# Document type classification keywords
_TYPE_HINTS: dict[str, List[str]] = {
    "contract": ["agreement", "contract", "party", "parties", "consideration", "whereas", "executed"],
    "notice": ["notice", "notification", "hereby notified", "demand", "cease and desist"],
    "case_file": ["plaintiff", "defendant", "court", "case no", "docket", "judgment", "order", "ruling"],
    "memo": ["memorandum", "memo", "internal", "re:", "subject:", "from:", "to:"],
    "title_document": ["deed", "title", "property", "parcel", "grantor", "grantee", "convey"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_dates(text: str) -> List[str]:
    found: set[str] = set()
    for pattern in _DATE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found.update(m.strip() for m in matches)
    # Sort and limit to avoid noise
    return sorted(found)[:20]


def _extract_case_numbers(text: str) -> List[str]:
    found: set[str] = set()
    for pattern in _CASE_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found.update(m.strip() for m in matches if len(m.strip()) >= 3)
    return sorted(found)[:10]


def _extract_parties(text: str) -> List[str]:
    found: set[str] = set()
    for pattern in _PARTY_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            cleaned = re.sub(r"\s+", " ", m).strip().rstrip(",;.")
            if 2 < len(cleaned) < 120:
                found.add(cleaned)
    return sorted(found)[:10]


def _extract_jurisdiction(text: str) -> Optional[str]:
    for pattern in _JURISDICTION_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().title()
    return None


def _extract_key_clauses(text: str) -> List[str]:
    """
    Identify clause headings that are present in the document.
    Returns the clause name + first 200 chars of the clause body.
    """
    found: List[str] = []
    upper = text.upper()
    for keyword in _CLAUSE_KEYWORDS:
        idx = upper.find(keyword)
        if idx != -1:
            # Grab up to 200 chars after the keyword as a snippet
            snippet = text[idx: idx + 200].replace("\n", " ").strip()
            found.append(snippet)
    return found[:15]


def _classify_type(text: str) -> str:
    lower = text.lower()
    scores: dict[str, int] = {}
    for doc_type, keywords in _TYPE_HINTS.items():
        scores[doc_type] = sum(lower.count(kw) for kw in keywords)
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "unknown"
    return best


# ── Public API ────────────────────────────────────────────────────────────────

def parse(extracted_pages: List[ExtractedText], filename: str) -> ParsedDocument:
    """
    Parse a list of extracted text pages into a structured ParsedDocument.

    Args:
        extracted_pages: Output from ocr.extract()
        filename:        Original filename (used for the doc record)

    Returns:
        ParsedDocument with all legal fields populated.
    """
    if not extracted_pages:
        log.warning("parse() called with empty page list for %s", filename)
        return ParsedDocument(
            doc_id=extracted_pages[0].doc_id if extracted_pages else str(uuid.uuid4()),
            filename=filename,
        )

    doc_id = extracted_pages[0].doc_id
    full_text = "\n\n".join(p.raw_text for p in extracted_pages if p.raw_text.strip())

    log.info("Parsing document %s (%d chars)", doc_id, len(full_text))

    doc_type = _classify_type(full_text)
    parties = _extract_parties(full_text)
    dates = _extract_dates(full_text)
    case_numbers = _extract_case_numbers(full_text)
    key_clauses = _extract_key_clauses(full_text)
    jurisdiction = _extract_jurisdiction(full_text)

    # Summary hint: first ~500 printable characters of the document
    summary_hint = re.sub(r"\s+", " ", full_text[:600]).strip()

    log.info(
        "Parsed %s: type=%s, parties=%d, dates=%d, cases=%d, clauses=%d",
        doc_id, doc_type, len(parties), len(dates), len(case_numbers), len(key_clauses),
    )

    return ParsedDocument(
        doc_id=doc_id,
        filename=filename,
        document_type=doc_type,
        parties=parties,
        dates=dates,
        case_numbers=case_numbers,
        key_clauses=key_clauses,
        jurisdiction=jurisdiction,
        summary_hint=summary_hint,
        full_text=full_text,
    )