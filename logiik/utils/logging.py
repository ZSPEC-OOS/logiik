"""
Logiik canonical logging module.
All other modules import from here — no inline print statements.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path

_LOG_DIR = Path("./logiik/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_formatter = logging.Formatter(
    fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

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
