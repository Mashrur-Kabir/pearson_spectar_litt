"""
test_parser.py — Unit tests for backend/parser.py
"""

import pytest
from backend.schemas import ExtractedText, ParsedDocument


def _page(text: str, doc_id: str = "doc1") -> ExtractedText:
    return ExtractedText(doc_id=doc_id, page_number=1, raw_text=text)


CONTRACT = """
This AGREEMENT entered on January 15, 2024 between:
Plaintiff: Smith & Associates LLC
Defendant: Johnson Corp.
Case No. 2024-CV-001.
WHEREAS the parties agree; NOW THEREFORE, in consideration:
CONFIDENTIALITY — All information is confidential.
GOVERNING LAW — State of New York.
TERMINATION — 30 days written notice.
IN THE UNITED STATES DISTRICT COURT FOR THE SOUTHERN DISTRICT OF NEW YORK.
"""

NOTICE = """
NOTICE OF DEMAND dated 15/03/2024.
Docket No. 2024-DN-555.
You are hereby notified that demand is made for payment.
"""


class TestDocumentType:
    def test_classifies_contract(self):
        from backend.parser import parse
        assert parse([_page(CONTRACT)], "f.pdf").document_type == "contract"

    def test_classifies_notice(self):
        from backend.parser import parse
        assert parse([_page(NOTICE)], "f.pdf").document_type == "notice"

    def test_unknown_for_blank(self):
        from backend.parser import _classify_type
        assert _classify_type("") == "unknown"


class TestDates:
    def test_iso_date(self):
        from backend.parser import _extract_dates
        assert any("2024" in d for d in _extract_dates("Date: 2024-01-15"))

    def test_slash_date(self):
        from backend.parser import _extract_dates
        assert len(_extract_dates("Filed 01/15/2024")) >= 1

    def test_written_date(self):
        from backend.parser import _extract_dates
        res = _extract_dates("Signed January 15, 2024")
        assert any("January" in d or "2024" in d for d in res)


class TestCaseNumbers:
    def test_case_no(self):
        from backend.parser import _extract_case_numbers
        result = _extract_case_numbers("Case No. 2024-CV-001")
        assert any("2024" in c for c in result)

    def test_docket_no(self):
        from backend.parser import _extract_case_numbers
        result = _extract_case_numbers("Docket No. 2024-DN-555")
        assert len(result) >= 1

    def test_no_false_positive(self):
        from backend.parser import _extract_case_numbers
        assert len(_extract_case_numbers("Hello world.")) == 0


class TestParties:
    def test_plaintiff(self):
        from backend.parser import _extract_parties
        result = _extract_parties("Plaintiff: Smith & Associates LLC\n")
        assert any("Smith" in p for p in result)

    def test_defendant(self):
        from backend.parser import _extract_parties
        result = _extract_parties("Defendant: Johnson Corp\n")
        assert any("Johnson" in p for p in result)


class TestKeyClauses:
    def test_finds_confidentiality(self):
        from backend.parser import _extract_key_clauses
        result = _extract_key_clauses(CONTRACT)
        assert any("CONFIDENTIALITY" in c.upper() for c in result)

    def test_finds_governing_law(self):
        from backend.parser import _extract_key_clauses
        result = _extract_key_clauses(CONTRACT)
        assert any("GOVERNING LAW" in c.upper() for c in result)

    def test_empty_text(self):
        from backend.parser import _extract_key_clauses
        assert _extract_key_clauses("") == []


class TestFullParse:
    def test_returns_parsed_document(self):
        from backend.parser import parse
        assert isinstance(parse([_page(CONTRACT)], "f.pdf"), ParsedDocument)

    def test_doc_id_propagated(self):
        from backend.parser import parse
        result = parse([_page(CONTRACT, doc_id="abc123")], "f.pdf")
        assert result.doc_id == "abc123"

    def test_full_text_populated(self):
        from backend.parser import parse
        result = parse([_page(CONTRACT)], "f.pdf")
        assert len(result.full_text) > 10

    def test_multi_page_concatenated(self):
        from backend.parser import parse
        pages = [_page("Page one text.", "d1"), _page("Page two text.", "d1")]
        result = parse(pages, "f.pdf")
        assert "Page one" in result.full_text and "Page two" in result.full_text

    def test_empty_list_returns_default(self):
        from backend.parser import parse
        result = parse([], "f.pdf")
        assert isinstance(result, ParsedDocument)