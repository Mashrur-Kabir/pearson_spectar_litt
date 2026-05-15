"""
retriever.py — Retrieval layer over ChromaDB.

Takes a natural-language query (and optional doc_id filter),
returns a RetrievalResult containing the most relevant chunks.

This is the single entry-point the generator uses for grounding.
"""

from __future__ import annotations

from typing import List, Optional

from backend.config import RETRIEVAL_TOP_K
from backend.embeddings import query_chunks
from backend.logger import get_logger
from backend.schemas import RetrievedChunk, RetrievalResult

log = get_logger(__name__)

# Chunks below this similarity score are discarded as noise
_MIN_SCORE: float = 0.05


def retrieve(
    query: str,
    doc_ids: Optional[List[str]] = None,
    top_k: int = RETRIEVAL_TOP_K,
    min_score: float = _MIN_SCORE,
) -> RetrievalResult:
    """
    Retrieve the most relevant chunks for a drafting task.

    Args:
        query:     The drafting task description or question.
        doc_ids:   Restrict search to these documents (None = search all).
        top_k:     Maximum number of chunks to return.
        min_score: Discard chunks with similarity below this threshold.

    Returns:
        RetrievalResult containing the filtered, deduplicated chunk list.
    """
    log.info(
        "Retrieving top-%d chunks | query='%.80s' | doc_ids=%s",
        top_k, query, doc_ids,
    )

    # Fetch more than needed so filtering doesn't leave us with too few
    raw: List[RetrievedChunk] = query_chunks(
        query=query,
        top_k=top_k,
        doc_ids=doc_ids,
    )

    # Filter out low-relevance chunks
    filtered = [c for c in raw if c.score >= min_score]

    # Deduplicate by chunk_id (defensive: ChromaDB should never return dups)
    seen: set[str] = set()
    deduped: List[RetrievedChunk] = []
    for chunk in filtered:
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            deduped.append(chunk)

    # Trim to top_k
    final = deduped[:top_k]

    log.info(
        "Retrieved %d chunks (raw=%d | after filter=%d | returned=%d).",
        len(final), len(raw), len(filtered), len(final),
    )

    return RetrievalResult(chunks=final, query=query)