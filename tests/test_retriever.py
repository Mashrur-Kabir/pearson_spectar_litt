"""
test_retriever.py — Unit tests for backend/retriever.py
"""

import pytest
from unittest.mock import patch

from backend.schemas import RetrievedChunk, RetrievalResult


def _chunk(cid: str, score: float = 0.85) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, doc_id="doc1", filename="test.pdf",
        chunk_index=0, text="Legal clause text.", score=score, page_number=1,
    )


class TestRetrieve:
    @patch("backend.retriever.query_chunks")
    def test_returns_retrieval_result(self, mock_q):
        mock_q.return_value = [_chunk("c1")]
        from backend.retriever import retrieve
        result = retrieve("confidentiality")
        assert isinstance(result, RetrievalResult)

    @patch("backend.retriever.query_chunks")
    def test_filters_low_scores(self, mock_q):
        mock_q.return_value = [_chunk("c1", 0.9), _chunk("c2", 0.03)]
        from backend.retriever import retrieve
        result = retrieve("test", min_score=0.1)
        assert all(c.score >= 0.1 for c in result.chunks)

    @patch("backend.retriever.query_chunks")
    def test_deduplicates_chunks(self, mock_q):
        mock_q.return_value = [_chunk("c1"), _chunk("c1")]
        from backend.retriever import retrieve
        result = retrieve("test")
        ids = [c.chunk_id for c in result.chunks]
        assert len(ids) == len(set(ids))

    @patch("backend.retriever.query_chunks")
    def test_respects_top_k(self, mock_q):
        mock_q.return_value = [_chunk(f"c{i}") for i in range(20)]
        from backend.retriever import retrieve
        result = retrieve("test", top_k=3)
        assert len(result.chunks) <= 3

    @patch("backend.retriever.query_chunks")
    def test_preserves_query_string(self, mock_q):
        mock_q.return_value = []
        from backend.retriever import retrieve
        result = retrieve("my exact query")
        assert result.query == "my exact query"

    @patch("backend.retriever.query_chunks")
    def test_empty_collection_returns_empty(self, mock_q):
        mock_q.return_value = []
        from backend.retriever import retrieve
        assert retrieve("test").chunks == []

    @patch("backend.retriever.query_chunks")
    def test_passes_doc_ids_to_query(self, mock_q):
        mock_q.return_value = []
        from backend.retriever import retrieve
        retrieve("test", doc_ids=["d1", "d2"])
        call_kwargs = mock_q.call_args[1]
        assert call_kwargs.get("doc_ids") == ["d1", "d2"]