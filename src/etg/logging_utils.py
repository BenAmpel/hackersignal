"""Structured logger with a Rich handler (falls back to stdlib on import error)."""

from __future__ import annotations

import logging

try:
    from rich.logging import RichHandler

    _RICH = True
except Exception:  # pragma: no cover
    _RICH = False


def get_logger(name: str = "etg", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    if _RICH:
        handler = RichHandler(show_time=True, show_path=False, markup=True)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
