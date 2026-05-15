"""
logger.py — Structured, coloured logging for the Legal AI System.

Every module gets its own named logger via get_logger(__name__).
Logs go to both the console (coloured) and a rotating file.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ── Log file location ─────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "legal_ai.log"

# ── Formatters ────────────────────────────────────────────────────────────────
_CONSOLE_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_FILE_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


class _ColourFormatter(logging.Formatter):
    """Adds ANSI colour codes to console output for quick visual scanning."""

    _COLOURS = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[35m",   # Magenta
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self._COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}{record.levelname}{self._RESET}"
        return super().format(record)


def _build_root_logger() -> logging.Logger:
    root = logging.getLogger("legal_ai")
    root.setLevel(logging.DEBUG)

    if root.handlers:
        # Already configured — avoid duplicate handlers on hot-reload
        return root

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(_ColourFormatter(_CONSOLE_FMT, datefmt=_DATE_FMT))

    # Rotating file handler (5 MB × 3 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))

    root.addHandler(console)
    root.addHandler(file_handler)
    return root


_ROOT = _build_root_logger()


def get_logger(name: str) -> logging.Logger:
    """
    Usage:
        from backend.logger import get_logger
        log = get_logger(__name__)
        log.info("something happened")
    """
    # Child loggers inherit level + handlers from _ROOT
    return _ROOT.getChild(name.replace("backend.", ""))