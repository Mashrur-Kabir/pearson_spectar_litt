"""
test_edit_learner.py — Unit tests for backend/edit_learner.py
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from backend.schemas import EditCapture, EditPattern


def _capture(original="The draft text here.", edited="**Revised Draft**\n\nThe draft text here."):
    return EditCapture(
        draft_id="draft-001",
        original_content=original,
        edited_content=edited,
        operator_notes="Added bold heading.",
    )


class TestDiffSummary:
    def test_no_diff_for_identical(self):
        from backend.edit_learner import _compute_diff_summary
        assert _compute_diff_summary("same", "same").strip() == ""

    def test_diff_shows_changes(self):
        from backend.edit_learner import _compute_diff_summary
        result = _compute_diff_summary("line one", "line two")
        assert "-" in result or "+" in result

    def test_truncates_at_limit(self):
        from backend.edit_learner import _compute_diff_summary
        a, b = "x " * 3000, "y " * 3000
        assert len(_compute_diff_summary(a, b)) <= 4200


class TestSimilarity:
    def test_identical_is_one(self):
        from backend.edit_learner import _similarity_ratio
        assert _similarity_ratio("hello", "hello") == 1.0

    def test_very_different_is_low(self):
        from backend.edit_learner import _similarity_ratio
        assert _similarity_ratio("aaa", "zzz") < 0.5

    def test_near_matches_are_high(self):
        from backend.edit_learner import _similarity_ratio
        assert _similarity_ratio(
            "Operator prefers bullet points",
            "Operator prefers bullet lists",
        ) > 0.7


class TestFindExisting:
    def test_finds_similar(self):
        from backend.edit_learner import _find_existing_pattern
        existing = [{"pattern_id": "p1", "draft_type": "case_fact_summary",
                     "description": "Operator prefers bullet points"}]
        result = _find_existing_pattern(
            "Operator prefers bullet lists", "case_fact_summary", existing, threshold=0.6
        )
        assert result == "p1"

    def test_wrong_type_no_match(self):
        from backend.edit_learner import _find_existing_pattern
        existing = [{"pattern_id": "p1", "draft_type": "notice_summary",
                     "description": "Operator prefers bullet points"}]
        assert _find_existing_pattern("Operator prefers bullet points",
                                      "case_fact_summary", existing) is None

    def test_below_threshold_no_match(self):
        from backend.edit_learner import _find_existing_pattern
        existing = [{"pattern_id": "p1", "draft_type": "case_fact_summary",
                     "description": "Something completely unrelated about fonts"}]
        assert _find_existing_pattern("Prefers footnotes for citations",
                                      "case_fact_summary", existing, threshold=0.95) is None


class TestCaptureAndLearn:
    @patch("backend.edit_learner.save_edit")
    @patch("backend.edit_learner.save_pattern")
    @patch("backend.edit_learner.get_patterns", return_value=[])
    @patch("backend.edit_learner._extract_pattern_via_llm")
    def test_returns_edit_pattern(self, mock_llm, mock_get, mock_save_pat, mock_save_edit):
        mock_llm.return_value = ("Operator adds bold headings.", "Use bold H2 headings.")
        from backend.edit_learner import capture_and_learn
        result = capture_and_learn(_capture(), "case_fact_summary")
        assert isinstance(result, EditPattern)
        assert result.description == "Operator adds bold headings."

    @patch("backend.edit_learner.save_edit")
    @patch("backend.edit_learner.increment_pattern_frequency")
    @patch("backend.edit_learner.get_patterns")
    @patch("backend.edit_learner._extract_pattern_via_llm")
    def test_increments_existing(self, mock_llm, mock_get, mock_inc, mock_save_edit):
        mock_llm.return_value = ("Operator adds bold headings.", "Use bold headings.")
        mock_get.return_value = [{
            "pattern_id": "existing-001",
            "draft_type": "case_fact_summary",
            "description": "Operator adds bold headings",
            "instruction": "Use bold headings.",
            "frequency": 2,
            "created_at": datetime.utcnow().isoformat(),
        }]
        from backend.edit_learner import capture_and_learn
        capture_and_learn(_capture(), "case_fact_summary")
        mock_inc.assert_called_once_with("existing-001")

    @patch("backend.edit_learner.save_edit")
    @patch("backend.edit_learner._extract_pattern_via_llm")
    def test_no_diff_skips_llm(self, mock_llm, mock_save_edit):
        from backend.edit_learner import capture_and_learn
        capture = EditCapture(draft_id="d1", original_content="same", edited_content="same")
        capture_and_learn(capture, "case_fact_summary")
        mock_llm.assert_not_called()