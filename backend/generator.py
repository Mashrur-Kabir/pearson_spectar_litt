"""
generator.py — Grounded draft generation using Google Gemini.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List

import google.generativeai as genai

from backend.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    MAX_OUTPUT_TOKENS,
    RETRIEVAL_TOP_K,
    TEMPERATURE,
)
from backend.db import get_patterns
from backend.grounding import build_grounded_prompt, format_evidence_block, is_well_grounded
from backend.logger import get_logger
from backend.retriever import retrieve
from backend.schemas import DraftRequest, DraftResponse

log = get_logger(__name__)

genai.configure(api_key=GEMINI_API_KEY)


def _get_model() -> genai.GenerativeModel:
    """Construct a Gemini GenerativeModel. Kept separate so tests can mock it."""
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=genai.GenerationConfig(
            max_output_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
        ),
    )


def _call_llm(prompt: str) -> str:
    """Call Gemini and return response text."""
    return _get_model().generate_content(prompt).text.strip()


def _load_learned_instructions(draft_type: str) -> List[str]:
    try:
        return [p["instruction"] for p in get_patterns(draft_type=draft_type)[:5]]
    except Exception as exc:
        log.warning("Could not load learned patterns: %s", exc)
        return []


def generate_draft(request: DraftRequest) -> DraftResponse:
    log.info(
        "Generating draft | type='%s' | model=%s",
        request.draft_type, GEMINI_MODEL,
    )

    # Step 1: Retrieve evidence
    retrieval = retrieve(
        query=request.query,
        doc_ids=request.doc_ids,
        top_k=RETRIEVAL_TOP_K,
    )

    warning = ""
    if not is_well_grounded(retrieval):
        log.warning("Weak evidence for query '%.60s'", request.query)
        warning = "⚠️  WARNING: Limited evidence retrieved. Draft may be incomplete.\n\n"

    # Step 2: Format evidence
    evidence_text, _ = format_evidence_block(retrieval.chunks)

    # Step 3: Load learned instructions
    learned = _load_learned_instructions(request.draft_type)
    if learned:
        log.info("Injecting %d learned instructions.", len(learned))

    # Step 4: Build prompt
    prompt = build_grounded_prompt(
        draft_type=request.draft_type,
        query=request.query,
        evidence_block=evidence_text,
        learned_instructions=learned,
    )
    log.debug("Prompt size: %d chars.", len(prompt))

    # Step 5: Call Gemini
    try:
        raw_content = _call_llm(prompt)
    except Exception as exc:
        log.error("Gemini API call failed: %s", exc)
        raise RuntimeError(f"Draft generation failed: {exc}") from exc

    if not raw_content:
        raise RuntimeError("Gemini returned an empty response.")

    return DraftResponse(
        draft_id=str(uuid.uuid4()),
        doc_ids=request.doc_ids,
        draft_type=request.draft_type,
        content=warning + raw_content,
        evidence=retrieval.chunks,
        generated_at=datetime.now(timezone.utc),
        model_used=GEMINI_MODEL,
    )