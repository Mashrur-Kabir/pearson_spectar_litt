"""
embeddings.py — ChromaDB vector store + sentence-transformer embeddings.

Responsibilities:
  - Initialize the ChromaDB persistent client and collection (once per process)
  - Embed text using sentence-transformers/all-MiniLM-L6-v2
  - Store Chunk objects into ChromaDB (with deduplication)
  - Query for nearest-neighbour chunks given a text query
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from backend.config import CHROMA_COLLECTION, CHROMA_DIR, EMBEDDING_MODEL
from backend.logger import get_logger
from backend.schemas import Chunk, RetrievedChunk

log = get_logger(__name__)

# ── Singletons (lazy-initialised once per process) ────────────────────────────
_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("Embedding model loaded.")
    return _model


def _get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(
            "ChromaDB collection '%s' ready. Total items: %d",
            CHROMA_COLLECTION,
            _collection.count(),
        )
    return _collection


# ── Public API ────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of strings with the sentence-transformer model.

    Args:
        texts: List of raw strings to embed.

    Returns:
        List of float vectors (one per input string).
    """
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, batch_size=32)
    return vectors.tolist()


def add_chunks(chunks: List[Chunk]) -> int:
    """
    Embed and store chunks in ChromaDB, skipping duplicates.

    Args:
        chunks: List of Chunk objects produced by the chunker.

    Returns:
        Number of chunks actually written (excludes duplicates).
    """
    if not chunks:
        return 0

    collection = _get_collection()

    # Check which IDs already exist so we don't re-embed unnecessarily
    try:
        existing = collection.get(ids=[c.chunk_id for c in chunks])
        existing_ids = set(existing["ids"])
    except Exception:
        existing_ids = set()

    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

    if not new_chunks:
        log.debug("All %d chunks already in ChromaDB — nothing to add.", len(chunks))
        return 0

    texts = [c.text for c in new_chunks]
    embeddings = embed_texts(texts)

    collection.add(
        ids=[c.chunk_id for c in new_chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "doc_id": c.doc_id,
                "chunk_index": str(c.chunk_index),
                "filename": str(c.metadata.get("filename", "")),
                "page_number": str(c.metadata.get("page_number", "1")),
            }
            for c in new_chunks
        ],
    )

    log.info(
        "Added %d chunks to ChromaDB (skipped %d existing).",
        len(new_chunks),
        len(chunks) - len(new_chunks),
    )
    return len(new_chunks)


def query_chunks(
    query: str,
    top_k: int = 5,
    doc_ids: Optional[List[str]] = None,
) -> List[RetrievedChunk]:
    """
    Find the most relevant chunks for a query string.

    Args:
        query:   The natural-language question or drafting task.
        top_k:   Maximum results to return.
        doc_ids: If given, restrict search to these document IDs only.

    Returns:
        List of RetrievedChunk ordered by relevance (highest score first).
    """
    collection = _get_collection()

    total = collection.count()
    if total == 0:
        log.warning("ChromaDB is empty — no chunks available to query.")
        return []

    query_embedding = embed_texts([query])[0]

    # Build optional metadata filter
    where: Optional[Dict[str, Any]] = None
    if doc_ids:
        if len(doc_ids) == 1:
            where = {"doc_id": doc_ids[0]}
        else:
            where = {"doc_id": {"$in": doc_ids}}

    n_results = min(top_k, total)

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        log.error("ChromaDB query failed: %s", e)
        return []

    retrieved: List[RetrievedChunk] = []
    for chunk_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # ChromaDB cosine distance: 0 = identical → similarity = 1 - dist/2
        score = max(0.0, 1.0 - dist / 2.0)
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                filename=meta.get("filename", ""),
                chunk_index=int(meta.get("chunk_index", 0)),
                text=doc,
                score=round(score, 4),
                page_number=int(meta["page_number"]) if meta.get("page_number") else None,
            )
        )

    top_score = retrieved[0].score if retrieved else 0.0
    log.debug("Query returned %d chunks (top score=%.3f).", len(retrieved), top_score)
    return retrieved


def delete_doc_chunks(doc_id: str) -> int:
    """
    Remove all chunks belonging to `doc_id` from ChromaDB.

    Returns:
        Number of chunks deleted.
    """
    collection = _get_collection()
    try:
        existing = collection.get(where={"doc_id": doc_id})
        ids_to_delete = existing["ids"]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            log.info("Deleted %d chunks for doc_id=%s.", len(ids_to_delete), doc_id)
        return len(ids_to_delete)
    except Exception as e:
        log.error("Failed to delete chunks for doc_id=%s: %s", doc_id, e)
        return 0


def health_check() -> bool:
    """Returns True if ChromaDB is reachable."""
    try:
        col = _get_collection()
        col.count()
        return True
    except Exception as e:
        log.error("ChromaDB health check failed: %s", e)
        return False