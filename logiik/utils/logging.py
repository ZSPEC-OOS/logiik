"""
Logiik canonical logging module.
All other modules import from here — no inline print statements.
"""
import logging
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path("./logiik/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_formatter = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# ── In-memory error buffer (WARNING and above) ────────────────────────────────

class _ErrorBufferHandler(logging.Handler):
    """Keeps the last 200 WARNING/ERROR/CRITICAL records in a ring buffer."""
    def __init__(self, maxlen: int = 200):
        super().__init__(level=logging.WARNING)
        self._buf: deque = deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord):
        self._buf.appendleft({
            "ts":      self.formatter.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        })

    def get_errors(self) -> list:
        return list(self._buf)

    def clear(self):
        self._buf.clear()


_error_buffer = _ErrorBufferHandler()
_error_buffer.setFormatter(_formatter)

# Attach to root logiik logger once
_root = logging.getLogger("logiik")
if not any(isinstance(h, _ErrorBufferHandler) for h in _root.handlers):
    _root.addHandler(_error_buffer)


def get_error_buffer() -> _ErrorBufferHandler:
    """Return the shared error buffer (used by the API endpoint)."""
    return _error_buffer


def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger writing to both stdout and a daily log file.
    Usage: logger = get_logger(__name__)
    """
    logger = logging.getLogger(f"logiik.{name}")
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_formatter)
    logger.addHandler(ch)

    # File handler — daily log file
    log_file = _LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_formatter)
    logger.addHandler(fh)

    return logger


def log_event(name: str, message: str, level: str = "info"):
    """
    Convenience function for one-line log calls from any module.
    Usage: log_event("training", "Phase 7 complete", level="info")
    """
    logger = get_logger(name)
    getattr(logger, level.lower(), logger.info)(message)
