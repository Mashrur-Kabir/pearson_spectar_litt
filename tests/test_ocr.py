"""
test_ocr.py — Unit tests for backend/ocr.py
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from backend.schemas import ExtractedText


class TestPreprocessing:
    def test_preprocess_returns_image(self):
        from backend.ocr import _preprocess_image

        img = Image.new("RGB", (100, 100), color="white")
        processed = _preprocess_image(img)

        assert processed is not None
        assert processed.mode == "L"


class TestCleanText:
    def test_clean_whitespace(self):
        from backend.ocr import _clean_text

        text = "Hello     world\n\n\nTest"
        result = _clean_text(text)

        assert "     " not in result
        assert "\n\n\n" not in result

    def test_remove_nonprintable(self):
        from backend.ocr import _clean_text

        text = "Hello\x00world"
        result = _clean_text(text)

        assert "\x00" not in result


class TestImageOCR:
    @patch("backend.ocr._ocr_image")
    def test_extract_image_success(self, mock_ocr, tmp_path):
        from backend.ocr import extract_image

        mock_ocr.return_value = ("Sample legal text", 0.95)

        img_path = tmp_path / "sample.png"
        Image.new("RGB", (100, 100), "white").save(img_path)

        result = extract_image(img_path, "doc1")

        assert len(result) == 1
        assert isinstance(result[0], ExtractedText)
        assert result[0].doc_id == "doc1"
        assert "Sample legal text" in result[0].raw_text

    def test_extract_invalid_image(self, tmp_path):
        from backend.ocr import extract_image

        bad = tmp_path / "bad.png"
        bad.write_text("not an image")

        with pytest.raises(RuntimeError):
            extract_image(bad, "doc1")


class TestRouting:
    @patch("backend.ocr.extract_pdf")
    def test_pdf_routing(self, mock_pdf):
        from backend.ocr import extract

        mock_pdf.return_value = []

        extract(Path("file.pdf"), "doc1")

        mock_pdf.assert_called_once()

    @patch("backend.ocr.extract_image")
    def test_image_routing(self, mock_img):
        from backend.ocr import extract

        mock_img.return_value = []

        extract(Path("img.png"), "doc1")

        mock_img.assert_called_once()

    def test_unsupported_file(self):
        from backend.ocr import extract

        with pytest.raises((ValueError, RuntimeError)):
            extract(Path("bad.xyz"), "doc1")


class TestOCRInternals:
    @patch("backend.ocr.pytesseract.image_to_data")
    def test_ocr_image_confidence(self, mock_data):
        from backend.ocr import _ocr_image

        mock_data.return_value = {
            "text": ["Legal", "Document"],
            "conf": ["95", "85"],
        }

        img = Image.new("RGB", (100, 100), "white")

        text, conf = _ocr_image(img)

        assert "Legal" in text
        assert conf > 0