"""
test_generator.py — Unit tests for backend/generator.py
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from backend.schemas import DraftRequest, DraftResponse, RetrievedChunk, RetrievalResult


def _retrieval(n: int = 3) -> RetrievalResult:
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}", doc_id="doc1", filename="doc.pdf",
            chunk_index=i, text=f"Evidence clause {i}.", score=0.9 - i * 0.05,
            page_number=i + 1,
        )
        for i in range(n)
    ]
    return RetrievalResult(chunks=chunks, query="Generate a summary.")


class TestGenerateDraft:
    @patch("backend.generator.get_patterns", return_value=[])
    @patch("backend.generator.retrieve")
    @patch("backend.generator._get_model")
    def test_returns_draft_response(self, mock_model, mock_retrieve, _):
        mock_retrieve.return_value = _retrieval(3)
        mock_model.return_value.generate_content.return_value = MagicMock(
            text="# Summary\n\nBased on [1] the parties are..."
        )
        from backend.generator import generate_draft
        result = generate_draft(DraftRequest(doc_ids=["doc1"], draft_type="case_fact_summary"))
        assert isinstance(result, DraftResponse)
        assert len(result.content) > 0

    @patch("backend.generator.get_patterns", return_value=[])
    @patch("backend.generator.retrieve")
    @patch("backend.generator._get_model")
    def test_evidence_attached(self, mock_model, mock_retrieve, _):
        mock_retrieve.return_value = _retrieval(2)
        mock_model.return_value.generate_content.return_value = MagicMock(text="Draft text.")
        from backend.generator import generate_draft
        result = generate_draft(DraftRequest(doc_ids=["doc1"], draft_type="internal_memo"))
        assert len(result.evidence) == 2

    @patch("backend.generator.get_patterns", return_value=[])
    @patch("backend.generator.retrieve")
    @patch("backend.generator._get_model")
    def test_no_evidence_adds_warning(self, mock_model, mock_retrieve, _):
        mock_retrieve.return_value = RetrievalResult(chunks=[], query="test")
        mock_model.return_value.generate_content.return_value = MagicMock(text="Weak draft.")
        from backend.generator import generate_draft
        result = generate_draft(DraftRequest(doc_ids=["doc1"], draft_type="notice_summary"))
        assert "WARNING" in result.content

    @patch("backend.generator.get_patterns", return_value=[])
    @patch("backend.generator.retrieve")
    @patch("backend.generator._get_model")
    def test_gemini_error_raises_runtime(self, mock_model, mock_retrieve, _):
        mock_retrieve.return_value = _retrieval(2)
        mock_model.return_value.generate_content.side_effect = Exception("API down")
        from backend.generator import generate_draft
        with pytest.raises(RuntimeError, match="Draft generation failed"):
            generate_draft(DraftRequest(doc_ids=["doc1"], draft_type="case_fact_summary"))

    @patch("backend.generator.get_patterns", return_value=[
        {"instruction": "Use bullet points for clause summaries."}
    ])
    @patch("backend.generator.retrieve")
    @patch("backend.generator._get_model")
    def test_learned_instructions_in_prompt(self, mock_model, mock_retrieve, _):
        mock_retrieve.return_value = _retrieval(2)
        mock_model.return_value.generate_content.return_value = MagicMock(text="Draft.")
        from backend.generator import generate_draft
        generate_draft(DraftRequest(doc_ids=["doc1"], draft_type="case_fact_summary"))
        # verify generate_content was called with a prompt containing "bullet"
        call_args = mock_model.return_value.generate_content.call_args
        prompt = call_args[0][0] if call_args[0] else str(call_args)
        assert "bullet" in prompt.lower()