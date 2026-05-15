"""
ocr.py — Text extraction for the Legal AI System.

EXTRACTION STRATEGY (in order of preference):
  1. pdfplumber       — fast and exact for digitally-born (text-layer) PDFs
  2. Gemini Vision    — primary OCR for scanned/image PDFs; reads cursive,
                        handwriting, stamps, mixed layouts far better than
                        Tesseract because it is a multimodal LLM
  3. pytesseract      — fallback if Gemini Vision call fails (printed text only)

WHY GEMINI VISION OVER TESSERACT FOR SCANNED DOCS:
  Tesseract is a classical OCR engine trained on printed fonts. It fails badly
  on cursive handwriting, historical scripts, mixed layouts, and degraded paper.
  Gemini 1.5 Flash is a vision-language model that genuinely understands
  what it is looking at, making it dramatically more accurate on the messy
  scanned legal documents this system is designed to handle.

Input:  file path (PDF or image)
Output: list of ExtractedText objects, one per page
"""

from __future__ import annotations

import base64
import io
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import google.generativeai as genai
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path

from backend.logger import get_logger
from backend.schemas import ExtractedText

log = get_logger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# Read directly from environment so this module stays self-contained
_GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
_GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
_OCR_DPI: int = int(os.getenv("OCR_DPI", "300"))
_TESSERACT_CMD: str = os.getenv("TESSERACT_CMD", "tesseract")

# Minimum characters pdfplumber must extract before we treat the page as a scan
_MIN_TEXT_CHARS = 50

# Configure Gemini SDK once at module load
if _GEMINI_API_KEY:
    genai.configure(api_key=_GEMINI_API_KEY)

pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

# ── Gemini Vision Prompt ──────────────────────────────────────────────────────
# This prompt instructs Gemini to act as a precise OCR engine rather than
# a summariser — we want every word preserved, not interpreted.
_GEMINI_OCR_PROMPT = """You are a precise document transcription engine.

Your task: transcribe ALL text visible in this document image with maximum accuracy.

CRITICAL RULES:
1. Transcribe EVERY word you can read — printed text, handwriting, stamps, and annotations.
2. For handwritten text: make your best effort even if uncertain. Never skip a field.
3. If a word is genuinely illegible, write [illegible] — but only as a last resort.
4. Preserve the document's structure: use newlines where the original has line breaks.
5. Do NOT summarise, interpret, or add commentary. Transcription only.
6. Include headers, form labels, filled-in values, dates, names, signatures, and stamps.
7. For fill-in-the-blank forms: include both the label AND the handwritten/typed value.

Begin transcription now:"""


# ── Image Preprocessing ───────────────────────────────────────────────────────

def _preprocess_for_tesseract(img: Image.Image) -> Image.Image:
    """
    Aggressive preprocessing pipeline for Tesseract (classical OCR).
    Tesseract needs clean, high-contrast, noise-free images.

    Steps:
    1. Convert to greyscale — removes colour noise
    2. Upscale 2× — Tesseract accuracy improves significantly at higher resolution
    3. Adaptive sharpening — improves character edges
    4. High contrast — makes ink stand out from aged/yellowed paper
    """
    # Step 1: Greyscale
    img = img.convert("L")

    # Step 2: Upscale 2× using LANCZOS (high quality)
    w, h = img.size
    img = img.resize((w * 2, h * 2), Image.LANCZOS)

    # Step 3: Sharpen
    img = img.filter(ImageFilter.SHARPEN)
    img = img.filter(ImageFilter.SHARPEN)  # double-sharpen for old docs

    # Step 4: High contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.5)

    return img


def _preprocess_for_gemini(img: Image.Image) -> Image.Image:
    """
    Lighter preprocessing for Gemini Vision.
    Gemini understands context, so we don't need aggressive binarisation.
    We only enhance enough to make the image clear for upload.
    """
    # Convert to RGB (Gemini prefers RGB over greyscale)
    img = img.convert("RGB")

    # Mild sharpening
    img = img.filter(ImageFilter.SHARPEN)

    # Mild contrast boost to make faded ink more visible
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)

    return img


def _image_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string for the Gemini API."""
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ── Gemini Vision OCR ─────────────────────────────────────────────────────────

def _ocr_with_gemini(img: Image.Image) -> Tuple[str, float]:
    """
    Use Gemini Vision to transcribe text from an image.

    This is the primary OCR path for scanned documents. Gemini can read:
    - Handwriting (including old cursive scripts)
    - Mixed printed + handwritten forms
    - Stamps and annotations
    - Degraded or faded text

    Returns:
        (transcribed_text, confidence)
        Confidence is 0.90 for a successful Gemini call (high trust),
        since Gemini validates its own reading contextually.
    """
    if not _GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Cannot use Gemini Vision for OCR. "
            "Set GEMINI_API_KEY in your .env file."
        )

    preprocessed = _preprocess_for_gemini(img)
    b64 = _image_to_base64(preprocessed)

    model = genai.GenerativeModel(model_name=_GEMINI_MODEL)

    # Send image + prompt to Gemini
    response = model.generate_content([
        _GEMINI_OCR_PROMPT,
        {
            "mime_type": "image/jpeg",
            "data": b64,
        },
    ])

    text = response.text.strip() if response.text else ""

    if not text:
        raise ValueError("Gemini Vision returned an empty transcription.")

    log.debug("Gemini Vision extracted %d chars.", len(text))
    return text, 0.90  # High confidence — Gemini Vision is reliable on these docs


# ── Tesseract OCR (Fallback) ──────────────────────────────────────────────────

def _ocr_with_tesseract(img: Image.Image) -> Tuple[str, float]:
    """
    Tesseract OCR fallback — used only when Gemini Vision fails.
    Works well for cleanly printed text; struggles with handwriting.

    Returns:
        (text, average_confidence)
    """
    preprocessed = _preprocess_for_tesseract(img)
    data = pytesseract.image_to_data(
        preprocessed,
        # psm 3: fully automatic page segmentation — better than 6 for mixed layouts
        # oem 3: use LSTM engine (most accurate for printed text)
        config=f"--dpi {_OCR_DPI} --oem 3 --psm 3",
        output_type=pytesseract.Output.DICT,
    )

    words: List[str] = []
    confidences: List[float] = []

    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        conf_int = int(conf)
        if word and conf_int > 0:
            words.append(word)
            confidences.append(conf_int / 100.0)

    text = " ".join(words)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    log.debug("Tesseract extracted %d chars, avg confidence %.2f", len(text), avg_conf)
    return text, avg_conf


# ── Smart OCR Router ──────────────────────────────────────────────────────────

def _run_ocr(img: Image.Image, page_num: int) -> Tuple[str, Optional[float]]:
    """
    Run OCR on a PIL Image using the best available method.

    Strategy:
    1. Try Gemini Vision first (best for handwriting, mixed layouts, old docs)
    2. If Gemini fails (API error, quota, etc.) → fall back to Tesseract
    3. If both fail → return empty string and log the error

    Args:
        img:      PIL Image of the page.
        page_num: Page number for logging only.

    Returns:
        (text, confidence)
    """
    # ── Attempt 1: Gemini Vision ──────────────────────────────────────────────
    if _GEMINI_API_KEY:
        try:
            log.info("Page %d: running Gemini Vision OCR", page_num)
            text, conf = _ocr_with_gemini(img)
            if text.strip():
                log.info("Page %d: Gemini Vision OK (%d chars)", page_num, len(text))
                return text, conf
            log.warning("Page %d: Gemini returned empty text, falling back to Tesseract", page_num)
        except Exception as gemini_err:
            log.warning(
                "Page %d: Gemini Vision failed (%s). Falling back to Tesseract.",
                page_num, gemini_err,
            )

    # ── Attempt 2: Tesseract ──────────────────────────────────────────────────
    try:
        log.info("Page %d: running Tesseract OCR", page_num)
        text, conf = _ocr_with_tesseract(img)
        log.info("Page %d: Tesseract OK (%d chars, conf=%.2f)", page_num, len(text), conf)
        return text, conf
    except Exception as tess_err:
        log.error("Page %d: Tesseract also failed: %s", page_num, tess_err)

    return "", None


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Light cleanup of extracted text.
    We deliberately keep this LIGHT — the LLM downstream handles interpretation.
    Over-aggressive cleaning destroys information.
    """
    # Collapse 3+ consecutive newlines to 2 (preserve paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces/tabs to a single space
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove genuinely non-printable characters (keep Unicode letters, punctuation)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", text)
    return text.strip()


# ── PDF Extraction ────────────────────────────────────────────────────────────

def extract_pdf(file_path: Path, doc_id: str) -> List[ExtractedText]:
    """
    Extract text from a PDF file.

    For each page:
      1. Try pdfplumber (fast, exact — works on text-layer PDFs)
      2. If the page yields too little text (it's a scanned image), run OCR

    Args:
        file_path: Path to the PDF file.
        doc_id:    Unique document identifier.

    Returns:
        List of ExtractedText objects, one per page.

    Common failure points:
      - pdfplumber raises if the PDF is corrupted or password-protected
      - pdf2image requires poppler to be installed (see README)
      - Gemini Vision requires a valid GEMINI_API_KEY
    """
    import pdfplumber

    results: List[ExtractedText] = []
    log.info("Extracting PDF: %s", file_path.name)

    # Pre-convert ALL pages to images in one pass (more efficient than per-page)
    # We'll use these images for any page that needs OCR
    page_images: dict[int, Image.Image] = {}
    try:
        images = convert_from_path(
            str(file_path),
            dpi=_OCR_DPI,
            fmt="jpeg",
        )
        for i, img in enumerate(images, start=1):
            page_images[i] = img
        log.debug("Pre-converted %d pages to images at %d DPI", len(images), _OCR_DPI)
    except Exception as img_err:
        # Non-fatal: we'll try pdfplumber only, then Tesseract from pdfplumber's image
        log.warning("Could not pre-convert PDF pages to images: %s", img_err)

    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):

                # ── Attempt 1: pdfplumber text layer ─────────────────────────
                text = page.extract_text() or ""
                confidence: Optional[float] = None
                extraction_method = "pdfplumber"

                if len(text.strip()) < _MIN_TEXT_CHARS:
                    # Page has no meaningful text layer — it's a scanned image
                    log.debug(
                        "Page %d: only %d chars from pdfplumber → OCR needed",
                        page_num, len(text.strip()),
                    )

                    # Get the page image (prefer pre-converted, fall back to pdfplumber)
                    img: Optional[Image.Image] = page_images.get(page_num)
                    if img is None:
                        try:
                            img = page.to_image(resolution=_OCR_DPI).original
                        except Exception as render_err:
                            log.warning(
                                "Page %d: could not render page image: %s",
                                page_num, render_err,
                            )

                    if img is not None:
                        text, confidence = _run_ocr(img, page_num)
                        extraction_method = (
                            "gemini_vision" if (confidence or 0) >= 0.89 else "tesseract"
                        )
                    else:
                        log.error("Page %d: no image available for OCR", page_num)

                cleaned = _clean_text(text)
                results.append(
                    ExtractedText(
                        doc_id=doc_id,
                        page_number=page_num,
                        raw_text=cleaned,
                        confidence=confidence,
                        extraction_method=extraction_method,
                    )
                )
                log.info(
                    "Page %d: %d chars via %s",
                    page_num, len(cleaned), extraction_method,
                )

    except Exception as e:
        log.error("Failed to extract PDF %s: %s", file_path.name, e)
        raise RuntimeError(f"PDF extraction failed for {file_path.name}: {e}") from e

    total_chars = sum(len(r.raw_text) for r in results)
    log.info(
        "PDF extraction complete: %d pages, %d total chars",
        len(results), total_chars,
    )
    return results


# ── Image Extraction ──────────────────────────────────────────────────────────

def extract_image(file_path: Path, doc_id: str) -> List[ExtractedText]:
    """
    Extract text from a single image file (jpg, png, tiff, etc.).

    Uses Gemini Vision first, Tesseract as fallback.
    """
    log.info("Extracting image: %s", file_path.name)
    try:
        img = Image.open(str(file_path))
        text, confidence = _run_ocr(img, page_num=1)
        cleaned = _clean_text(text)
        return [
            ExtractedText(
                doc_id=doc_id,
                page_number=1,
                raw_text=cleaned,
                confidence=confidence,
                extraction_method=(
                    "gemini_vision" if (confidence or 0) >= 0.89 else "tesseract"
                ),
            )
        ]
    except Exception as e:
        log.error("Image extraction failed for %s: %s", file_path.name, e)
        raise RuntimeError(f"Image extraction failed for {file_path.name}: {e}") from e


# ── Public API ────────────────────────────────────────────────────────────────

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def extract(file_path: Path, doc_id: str) -> List[ExtractedText]:
    """
    Route to the right extractor based on file extension.

    Args:
        file_path: Path to the uploaded file.
        doc_id:    Unique document identifier.

    Returns:
        List of ExtractedText objects, one per page / image.

    Raises:
        ValueError:   If file type is not supported.
        RuntimeError: If extraction fails entirely.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(file_path, doc_id)
    elif suffix in SUPPORTED_IMAGE_EXTS:
        return extract_image(file_path, doc_id)
    else:
        # Last resort: try the unstructured library
        try:
            from unstructured.partition.auto import partition
            elements = partition(filename=str(file_path))
            combined = "\n\n".join(str(el) for el in elements if str(el).strip())
            cleaned = _clean_text(combined)
            return [
                ExtractedText(
                    doc_id=doc_id,
                    page_number=1,
                    raw_text=cleaned,
                    confidence=None,
                    extraction_method="unstructured",
                )
            ]
        except ImportError:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: .pdf, {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Extraction failed for {file_path.name}: {e}"
            ) from e