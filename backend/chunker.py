"""
chunker.py — Split legal document text into overlapping chunks for embedding.

Why chunking?
  Embedding models have a token limit (typically 512 tokens).
  Long legal documents must be split into smaller pieces so that
  each piece fits in one embedding and is semantically focused.

Strategy: character-level sliding window with sentence-boundary awareness.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from typing import List

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP
from backend.logger import get_logger
from backend.schemas import Chunk, ParsedDocument

log = get_logger(__name__)


def _split_into_sentences(text: str) -> List[str]:
    """
    A lightweight sentence splitter that handles:
      - Standard sentence endings (. ! ?)
      - Legal numbering (1. 2. (a) (i))
      - Newlines as natural break points
    """
    # Split on sentence boundaries, keeping the delimiter
    parts = re.split(r"(?<=[.!?])\s+|(?<=\n)\s*(?=\n)", text)
    # Filter out empty strings and strip whitespace
    return [p.strip() for p in parts if p.strip()]


def _make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Deterministic chunk ID based on content — stable across re-ingestion."""
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}_{chunk_index:04d}_{content_hash}"


def chunk_text(
    doc_id: str,
    text: str,
    filename: str,
    page_number: int = 1,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Split `text` into overlapping chunks of approximately `chunk_size` characters.

    The splitter tries to break on sentence boundaries so chunks are
    semantically coherent rather than mid-sentence.

    Args:
        doc_id:       Document identifier.
        text:         The full text to split.
        filename:     Original filename (stored in metadata).
        page_number:  Page this text came from (stored in metadata).
        chunk_size:   Target maximum characters per chunk.
        chunk_overlap: Characters to re-include from the previous chunk.

    Returns:
        List of Chunk objects ready to be embedded and stored.
    """
    if not text.strip():
        return []

    sentences = _split_into_sentences(text)
    chunks: List[Chunk] = []
    current: List[str] = []
    current_len = 0
    chunk_index = 0

    for sentence in sentences:
        sent_len = len(sentence)

        # If a single sentence is bigger than the chunk size, split it hard
        if sent_len > chunk_size:
            # Flush current buffer first
            if current:
                chunk_text_str = " ".join(current)
                chunk_id = _make_chunk_id(doc_id, chunk_index, chunk_text_str)
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    text=chunk_text_str,
                    metadata={"filename": filename, "page_number": page_number},
                ))
                chunk_index += 1
                current = []
                current_len = 0

            # Hard-split the long sentence
            for i in range(0, sent_len, chunk_size - chunk_overlap):
                piece = sentence[i: i + chunk_size]
                if piece.strip():
                    c_id = _make_chunk_id(doc_id, chunk_index, piece)
                    chunks.append(Chunk(
                        chunk_id=c_id,
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        text=piece,
                        metadata={"filename": filename, "page_number": page_number},
                    ))
                    chunk_index += 1
            continue

        # Normal case: accumulate sentences
        if current_len + sent_len + 1 > chunk_size and current:
            # Flush current buffer
            chunk_text_str = " ".join(current)
            c_id = _make_chunk_id(doc_id, chunk_index, chunk_text_str)
            chunks.append(Chunk(
                chunk_id=c_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
                text=chunk_text_str,
                metadata={"filename": filename, "page_number": page_number},
            ))
            chunk_index += 1

            # Overlap: re-include the last N chars of the previous chunk
            overlap_text = chunk_text_str[-chunk_overlap:] if chunk_overlap else ""
            current = [overlap_text] if overlap_text.strip() else []
            current_len = len(overlap_text)

        current.append(sentence)
        current_len += sent_len + 1  # +1 for the space

    # Flush whatever is left
    if current:
        chunk_text_str = " ".join(current)
        if chunk_text_str.strip():
            c_id = _make_chunk_id(doc_id, chunk_index, chunk_text_str)
            chunks.append(Chunk(
                chunk_id=c_id,
                doc_id=doc_id,
                chunk_index=chunk_index,
                text=chunk_text_str,
                metadata={"filename": filename, "page_number": page_number},
            ))

    log.debug(
        "Chunked doc %s page %d → %d chunks (avg %.0f chars)",
        doc_id, page_number, len(chunks),
        sum(len(c.text) for c in chunks) / max(len(chunks), 1),
    )
    return chunks


def chunk_document(parsed: ParsedDocument) -> List[Chunk]:
    """
    Chunk a fully-parsed document.
    Splits the full_text into page-sized sections first, then chunks each section.

    Args:
        parsed: ParsedDocument from parser.parse()

    Returns:
        Flat list of all chunks across all pages.
    """
    all_chunks: List[Chunk] = []

    if not parsed.full_text.strip():
        log.warning("chunk_document: empty full_text for %s", parsed.doc_id)
        return all_chunks

    # Split by double-newline as a page/section proxy
    sections = re.split(r"\n{2,}", parsed.full_text)

    for page_num, section in enumerate(sections, start=1):
        section = section.strip()
        if not section:
            continue
        chunks = chunk_text(
            doc_id=parsed.doc_id,
            text=section,
            filename=parsed.filename,
            page_number=page_num,
        )
        all_chunks.extend(chunks)

    log.info(
        "Document %s → %d total chunks from %d sections",
        parsed.doc_id, len(all_chunks), len(sections),
    )
    return all_chunks