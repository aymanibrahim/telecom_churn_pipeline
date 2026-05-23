"""Centralised logging configuration.

Every module in ``src`` and every script grabs its logger via::

    from src.logging_setup import get_logger
    log = get_logger(__name__)

Output goes to stdout by default. Inside Airflow, the task logger
captures stdout into the per-task log file automatically.

Set ``CHURN_LOG_LEVEL=DEBUG`` in the environment to turn on debug logs.
"""
from __future__ import annotations

import logging
import os
import sys

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_CONFIGURED: bool = False


def _configure_root() -> None:
    """Idempotently configure the root logger with stdout handler.

    Inside a framework that already configures logging (Airflow, pytest,
    Streamlit) the root logger already has handlers, and adding another
    StreamHandler that writes to stdout creates an infinite recursion when
    the framework captures stdout and re-routes it back through logging.
    So we detect that case and just respect the framework's setup.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = os.environ.get("CHURN_LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()

    if root.handlers:
        # Framework (Airflow / pytest / etc.) already configured logging.
        # Don't add a competing handler; just set our package loggers' level
        # and let propagation handle output.
        for name in ("src", "churn", "scripts"):
            logging.getLogger(name).setLevel(level)
        _CONFIGURED = True
        return

    # Standalone CLI mode — configure from scratch.
    # Force UTF-8 on stdout so log messages with Unicode (arrows, accents)
    # don't blow up on Windows cp1252 consoles.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    root.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    handler._churn = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    # Tame chatty third-parties.
    for noisy in ("dask", "distributed", "matplotlib", "fiona", "rasterio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger. Safe to call from any module."""
    _configure_root()
    return logging.getLogger(name or "churn")
