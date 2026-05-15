"""
edit_learner.py — Learn reusable editing patterns from operator corrections.

Full flow:
  1. Save the raw edit (original vs edited text) to SQLite.
  2. Compute a unified diff between original and edited.
  3. If there is a meaningful diff, send it to Gemini and ask for
     a generalizable instruction.
  4. Deduplicate against existing patterns using string similarity.
  5. Save a new pattern or increment frequency on an existing one.
  6. Return the EditPattern.
"""

from __future__ import annotations

import difflib
import uuid
from datetime import datetime
from typing import List, Optional, Tuple

import google.generativeai as genai

from backend.config import GEMINI_API_KEY, GEMINI_MODEL, MAX_OUTPUT_TOKENS, TEMPERATURE
from backend.db import (
    get_patterns,
    increment_pattern_frequency,
    save_edit,
    save_pattern,
)
from backend.logger import get_logger
from backend.schemas import EditCapture, EditPattern

log = get_logger(__name__)

genai.configure(api_key=GEMINI_API_KEY)


# ── Diff helpers ──────────────────────────────────────────────────────────────

def _compute_diff_summary(original: str, edited: str, max_chars: int = 4000) -> str:
    """
    Produce a unified diff string between two texts, capped at max_chars.
    Lines starting with '-' are removals; '+' are additions.
    """
    orig_lines = original.splitlines(keepends=True)
    edit_lines = edited.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            orig_lines, edit_lines,
            fromfile="original_draft",
            tofile="edited_draft",
            lineterm="",
            n=2,
        )
    )
    diff_text = "".join(diff)
    if len(diff_text) > max_chars:
        diff_text = diff_text[:max_chars] + "\n... (diff truncated)"
    return diff_text


def _similarity_ratio(a: str, b: str) -> float:
    """Return a 0-1 similarity score between two strings."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ── Pattern extraction via Gemini ─────────────────────────────────────────────

def _extract_pattern_via_llm(
    draft_type: str,
    diff_summary: str,
    operator_notes: Optional[str],
) -> Tuple[str, str]:
    """
    Ask Gemini to analyse the diff and return ONE generalizable edit pattern.

    Returns:
        (description, instruction)
    """
    notes_line = f"\nOperator's own notes: {operator_notes}" if operator_notes else ""

    prompt = f"""You are reviewing edits an operator made to an AI-generated legal draft.
Draft type: {draft_type}{notes_line}

The diff below shows what was changed (lines starting with '-' were removed, '+' were added):

{diff_summary}

Identify the single most generalizable editing preference shown by these changes.

Respond in EXACTLY this format — nothing else, no preamble, no markdown:

DESCRIPTION: <one concise sentence describing the pattern>
INSTRUCTION: <one actionable sentence to inject into future generation prompts>
"""

    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=genai.GenerationConfig(
                max_output_tokens=MAX_OUTPUT_TOKENS,
                temperature=TEMPERATURE,
            ),
        )
        response = model.generate_content(prompt)
        text = response.text.strip()

        description = ""
        instruction = ""
        for line in text.splitlines():
            if line.startswith("DESCRIPTION:"):
                description = line.removeprefix("DESCRIPTION:").strip()
            elif line.startswith("INSTRUCTION:"):
                instruction = line.removeprefix("INSTRUCTION:").strip()

        if not description or not instruction:
            raise ValueError(f"Unexpected LLM format: {text!r}")

        return description, instruction

    except Exception as exc:
        log.warning("Pattern extraction failed (%s) — using fallback.", exc)
        return (
            "Operator made structural or stylistic edits to the draft.",
            "Ensure the draft is well-structured, comprehensive, and clearly formatted.",
        )


# ── Deduplication ─────────────────────────────────────────────────────────────

def _find_existing_pattern(
    description: str,
    draft_type: str,
    existing: List[dict],
    threshold: float = 0.75,
) -> Optional[str]:
    """
    Return the pattern_id of a sufficiently similar existing pattern, or None.
    """
    for pat in existing:
        if pat.get("draft_type") != draft_type:
            continue
        sim = _similarity_ratio(description, pat["description"])
        if sim >= threshold:
            log.debug("Pattern similarity %.2f — match: %s", sim, pat["pattern_id"])
            return pat["pattern_id"]
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def capture_and_learn(capture: EditCapture, draft_type: str) -> EditPattern:
    """
    Process an operator edit and update the pattern library.

    Args:
        capture:    EditCapture with original + edited content.
        draft_type: The draft type being edited.

    Returns:
        The created or updated EditPattern.
    """
    log.info("Processing operator edit for draft_id=%s", capture.draft_id)

    # 1. Persist the raw edit
    edit_id = str(uuid.uuid4())
    save_edit(edit_id, capture)

    # 2. Compute diff
    diff = _compute_diff_summary(capture.original_content, capture.edited_content)

    if not diff.strip():
        log.info("No meaningful diff for edit_id=%s — no pattern extracted.", edit_id)
        now = datetime.utcnow()
        return EditPattern(
            pattern_id=str(uuid.uuid4()),
            description="No significant changes detected in this edit.",
            instruction="Maintain current draft structure and style.",
            draft_type=draft_type,
            frequency=1,
            created_at=now,
            last_seen_at=now,
        )

    # 3. Extract pattern via Gemini
    description, instruction = _extract_pattern_via_llm(
        draft_type=draft_type,
        diff_summary=diff,
        operator_notes=capture.operator_notes,
    )
    log.info("Pattern extracted: '%s'", description)

    # 4. Check for existing similar pattern
    existing_patterns = get_patterns(draft_type=draft_type)
    existing_id = _find_existing_pattern(description, draft_type, existing_patterns)

    now = datetime.utcnow()

    if existing_id:
        increment_pattern_frequency(existing_id)
        log.info("Existing pattern updated: %s (frequency +1).", existing_id)
        for pat in existing_patterns:
            if pat["pattern_id"] == existing_id:
                return EditPattern(
                    pattern_id=pat["pattern_id"],
                    description=pat["description"],
                    instruction=pat["instruction"],
                    draft_type=pat["draft_type"],
                    frequency=pat["frequency"] + 1,
                    created_at=datetime.fromisoformat(pat["created_at"]),
                    last_seen_at=now,
                )

    # 5. Save new pattern
    new_pattern = EditPattern(
        pattern_id=str(uuid.uuid4()),
        description=description,
        instruction=instruction,
        draft_type=draft_type,
        frequency=1,
        created_at=now,
        last_seen_at=now,
    )
    save_pattern(new_pattern)
    log.info("New pattern saved: %s", new_pattern.pattern_id)

    return new_pattern