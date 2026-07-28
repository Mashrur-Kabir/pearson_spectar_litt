"""
embeddings.py — Adaptive vector store for the Legal AI System.

When PINECONE_API_KEY is set:  uses Pinecone Serverless [Production].
When it is NOT set:            falls back to local ChromaDB [Local Dev].

Public API (same regardless of backend):
  embed_texts(texts)           → List[List[float]]
  add_chunks(chunks)           → int  (count written)
  query_chunks(query, ...)     → List[RetrievedChunk]
  delete_doc_chunks(doc_id)    → int  (count removed)
  health_check()               → bool
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import time

from sentence_transformers import SentenceTransformer

from backend.config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME,
    EMBEDDING_MODEL, CHROMA_COLLECTION,
)
from backend.logger import get_logger
from backend.schemas import Chunk, RetrievedChunk

log = get_logger(__name__)

# ── Backend selection ──────────────────────────────────────────────────────────

USE_PINECONE = bool(PINECONE_API_KEY)

if USE_PINECONE:
    from pinecone import Pinecone as _Pinecone
    log.info("Using Pinecone as the vector store.")
else:
    import chromadb
    log.warning("PINECONE_API_KEY not set — falling back to local ChromaDB.")

# ── Singletons ─────────────────────────────────────────────────────────────────

_model: Optional[SentenceTransformer] = None
_pinecone_index: Any = None
_chroma_collection: Any = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("Embedding model loaded.")
    return _model


def _get_pinecone_index() -> Any:
    global _pinecone_index
    if _pinecone_index is None:
        pc = _Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        log.info("Pinecone index '%s' ready.", PINECONE_INDEX_NAME)
    return _pinecone_index


def _get_chroma_collection() -> Any:
    global _chroma_collection
    if _chroma_collection is None:
        client = chromadb.PersistentClient(path="./data/chroma")
        _chroma_collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("ChromaDB collection '%s' ready. Total items: %d",
                 CHROMA_COLLECTION, _chroma_collection.count())
    return _chroma_collection


# ── Public API ─────────────────────────────────────────────────────────────────

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, batch_size=32)
    return vectors.tolist()


def add_chunks(chunks: List[Chunk]) -> int:
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    if USE_PINECONE:
        index = _get_pinecone_index()
        vectors_to_upsert = []
        for c, emb in zip(chunks, embeddings):
            vectors_to_upsert.append({
                "id": c.chunk_id,
                "values": emb,
                "metadata": {
                    "doc_id": c.doc_id,
                    "chunk_index": c.chunk_index,
                    "filename": str(c.metadata.get("filename", "")),
                    "page_number": int(c.metadata.get("page_number", 1)),
                    "text": c.text,
                }
            })
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            index.upsert(vectors=vectors_to_upsert[i: i + batch_size])
            time.sleep(0.1)
        log.info("Added %d chunks to Pinecone index '%s'.", len(chunks), PINECONE_INDEX_NAME)
    else:
        col = _get_chroma_collection()
        col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "filename": str(c.metadata.get("filename", "")),
                "page_number": int(c.metadata.get("page_number", 1)),
            } for c in chunks],
        )
        log.info("Added %d chunks to ChromaDB collection '%s'.", len(chunks), CHROMA_COLLECTION)

    return len(chunks)


def query_chunks(
    query: str,
    top_k: int = 5,
    doc_ids: Optional[List[str]] = None,
) -> List[RetrievedChunk]:
    query_embedding = embed_texts([query])[0]

    if USE_PINECONE:
        index = _get_pinecone_index()
        where: Optional[Dict[str, Any]] = None
        if doc_ids:
            where = {"doc_id": {"$eq": doc_ids[0]}} if len(doc_ids) == 1 \
                else {"doc_id": {"$in": doc_ids}}
        try:
            results = index.query(
                vector=query_embedding, top_k=top_k,
                filter=where, include_metadata=True
            )
        except Exception as e:
            log.error("Pinecone query failed: %s", e)
            return []

        retrieved = []
        for match in results.get("matches", []):
            meta = match.get("metadata", {})
            retrieved.append(RetrievedChunk(
                chunk_id=match["id"], doc_id=meta.get("doc_id", ""),
                filename=meta.get("filename", ""), chunk_index=int(meta.get("chunk_index", 0)),
                text=meta.get("text", ""), score=round(match.get("score", 0.0), 4),
                page_number=int(meta.get("page_number", 1))
            ))
    else:
        col = _get_chroma_collection()
        where_chroma = {"doc_id": {"$in": doc_ids}} if doc_ids else None
        try:
            results = col.query(
                query_embeddings=[query_embedding], n_results=min(top_k, col.count() or 1),
                where=where_chroma, include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            log.error("ChromaDB query failed: %s", e)
            return []

        retrieved = []
        if results and results.get("ids"):
            for idx, chunk_id in enumerate(results["ids"][0]):
                meta = results["metadatas"][0][idx]
                dist = results["distances"][0][idx]
                retrieved.append(RetrievedChunk(
                    chunk_id=chunk_id, doc_id=meta.get("doc_id", ""),
                    filename=meta.get("filename", ""), chunk_index=int(meta.get("chunk_index", 0)),
                    text=results["documents"][0][idx], score=round(1 - dist, 4),
                    page_number=int(meta.get("page_number", 1))
                ))

    log.debug("Query returned %d chunks.", len(retrieved))
    return retrieved


def delete_doc_chunks(doc_id: str) -> int:
    if USE_PINECONE:
        try:
            _get_pinecone_index().delete(filter={"doc_id": {"$eq": doc_id}})
            log.info("Deleted chunks for doc_id=%s from Pinecone.", doc_id)
            return 1
        except Exception as e:
            log.error("Pinecone delete failed for doc_id=%s: %s", doc_id, e)
            return 0
    else:
        try:
            col = _get_chroma_collection()
            existing = col.get(where={"doc_id": doc_id})
            if existing["ids"]:
                col.delete(ids=existing["ids"])
                log.info("Deleted %d chunks for doc_id=%s from ChromaDB.", len(existing["ids"]), doc_id)
                return len(existing["ids"])
            return 0
        except Exception as e:
            log.error("ChromaDB delete failed for doc_id=%s: %s", doc_id, e)
            return 0


def health_check() -> bool:
    try:
        if USE_PINECONE:
            _get_pinecone_index().describe_index_stats()
        else:
            _get_chroma_collection().count()
        return True
    except Exception as e:
        log.error("Vector store health check failed: %s", e)
        return False