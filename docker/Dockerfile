# ── Legal AI System — Dockerfile ─────────────────────────────────────────────
# Python 3.12 slim base for a lean image

FROM python:3.12-slim

# ── System dependencies ───────────────────────────────────────────────────────
# tesseract-ocr: OCR engine
# poppler-utils: PDF→image conversion for OCR fallback
# libgl1: OpenCV/PIL runtime dependency
# libglib2.0-0: shared library for several packages

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Create data directories ───────────────────────────────────────────────────
RUN mkdir -p data/uploads data/chroma_db

# ── Environment defaults (override via docker-compose or -e flags) ────────────
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV TESSERACT_CMD=tesseract
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Expose FastAPI port ───────────────────────────────────────────────────────
EXPOSE 8000
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]