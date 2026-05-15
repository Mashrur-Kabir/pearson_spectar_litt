"""
grounding.py — Evidence formatting and grounded prompt assembly.

This module does three things:
  1. Converts a list of RetrievedChunk objects into a human-readable
     numbered evidence block for injection into LLM prompts.
  2. Checks whether there is enough evidence to justify generation.
  3. Assembles the final prompt that constrains the LLM to only use
     the provided evidence (no hallucination).
"""

from __future__ import annotations

from typing import List, Tuple

from backend.logger import get_logger
from backend.schemas import RetrievedChunk, RetrievalResult

log = get_logger(__name__)

# We need at least this many chunks to consider the context grounded
_MIN_GROUNDED_CHUNKS: int = 1


def format_evidence_block(chunks: List[RetrievedChunk]) -> Tuple[str, List[str]]:
    """
    Format retrieved chunks into a numbered evidence block.

    Args:
        chunks: List of RetrievedChunk from the retriever.

    Returns:
        (evidence_text, citation_labels)
        - evidence_text  : Multi-line string ready to inject into an LLM prompt.
        - citation_labels: Short labels like "[1] filename.pdf p.3 (0.92)"
    """
    if not chunks:
        return "No evidence retrieved from source documents.", []

    lines: List[str] = []
    labels: List[str] = []

    for i, chunk in enumerate(chunks, start=1):
        page_str = f" p.{chunk.page_number}" if chunk.page_number else ""
        label = f"[{i}] {chunk.filename}{page_str} (relevance: {chunk.score:.2f})"
        labels.append(label)

        lines.append(f"--- Evidence [{i}] ---")
        lines.append(f"Source: {chunk.filename}{page_str}")
        lines.append(chunk.text.strip())
        lines.append("")

    return "\n".join(lines).strip(), labels


def is_well_grounded(result: RetrievalResult) -> bool:
    """
    Returns True when there is enough evidence to proceed with grounded generation.
    """
    return len(result.chunks) >= _MIN_GROUNDED_CHUNKS


def build_grounded_prompt(
    draft_type: str,
    query: str,
    evidence_block: str,
    learned_instructions: list,
) -> str:
    _type_descriptions = {
        "case_fact_summary":  "a structured case fact summary",
        "title_review":       "a title review summary",
        "notice_summary":     "a notice-related summary",
        "document_checklist": "a document checklist",
        "internal_memo":      "a first-pass internal legal memo",
    }
    output_desc = _type_descriptions.get(draft_type, "a legal draft")

    learned_block = ""
    if learned_instructions:
        lines = "\n".join(f"  - {inst}" for inst in learned_instructions)
        learned_block = f"\n\nADDITIONAL FORMATTING INSTRUCTIONS (from past operator feedback):\n{lines}"

    prompt = f"""You are a senior legal analyst producing {output_desc} for internal review.

CRITICAL EXTRACTION RULES AND INSTRUCTIONS — READ BEFORE WRITING:
1. Extract and summarize ALL factual information present in the evidence below.
2. Cite evidence inline using [N] where N is the evidence number.
3. Only write "Not found in source documents." if a section truly has zero relevant evidence.
4. Do NOT invent facts not present in the evidence.
5. Do NOT include sections that are irrelevant to the document type — if a section has no applicable information AND no evidence, omit it entirely rather than writing "Not found."
6. Adapt your section headings to match the actual document being reviewed, not a generic template.
7. Use clear section headings and professional language.
8. IMPORTANT — handling partially legible text: The evidence may contain OCR artifacts or partially unclear text from scanned documents.
9. If text appears ambiguous but reasonably inferable, make your best reading and optionally add a note like "(OCR: possibly X)".
10. Only mark text as [illegible] if it is truly unreadable.
11. Never invent facts, dates, names, or clauses that are not supported by evidence.{learned_block}

DRAFTING TASK:
{query}

SOURCE EVIDENCE (read ALL of it carefully before writing):
{evidence_block}

---
Produce the {output_desc} now. Only include sections relevant to this specific document.
"""
    return prompt.strip()