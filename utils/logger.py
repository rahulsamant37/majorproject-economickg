"""
Structured logging utility for the application.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache


def _create_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)-28s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@lru_cache(maxsize=32)
def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with consistent formatting.

    Uses lru_cache so repeated calls for the same name return the
    same logger instance (avoiding duplicate handlers).
    """
    from config.settings import get_settings

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_create_formatter())
        logger.addHandler(handler)

    logger.setLevel(get_settings().log_level.upper())
    return logger
